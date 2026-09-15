# -*- coding: utf-8 -*-
"""
Calculadora de Soluciones Nutritivas para Fertirriego bajo Ambiente Protegido
===============================================================================
Aplicación Streamlit para el cálculo de soluciones nutritivas, neutralización
de bicarbonatos y dosificación de fertilizantes en Tanques Madre (A, B y C).

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# 0. CONFIGURACIÓN GENERAL Y CONSTANTES
# =============================================================================

st.set_page_config(
    page_title="Calculadora de Fertirriego",
    page_icon="🌱",
    layout="wide",
)

# Nutrientes elementales manejados por la app (todos en base elemental, ppm)
NUTRIENTES = ["N", "P", "K", "Ca", "Mg", "S", "Fe", "Zn", "B"]
NOMBRES_NUTRIENTES = {
    "N": "Nitrógeno (N)",
    "P": "Fósforo (P)",
    "K": "Potasio (K)",
    "Ca": "Calcio (Ca)",
    "Mg": "Magnesio (Mg)",
    "S": "Azufre (S)",
    "Fe": "Hierro (Fe)",
    "Zn": "Zinc (Zn)",
    "B": "Boro (B)",
}

# Factores de conversión de formas iónicas / óxidos a elemento puro
F_NO3_A_N = 14.0 / 62.0     # NO3- -> N
F_SO4_A_S = 32.0 / 96.0     # SO4-2 -> S
F_PO4_A_P = 31.0 / 95.0     # PO4-3 -> P  (referencia, no usado por defecto)
F_P2O5_A_P = (2 * 31.0) / 142.0   # P2O5 -> P
F_K2O_A_K = (2 * 39.1) / 94.2     # K2O -> K
HCO3_EQ = 61.0               # peso equivalente del HCO3- (meq/L = ppm/61)

# Pesos equivalentes de los ácidos (mg por meq == mg por mmol, 1 eq = 1 H+ útil
# en el rango de pH de neutralización de bicarbonatos)
PESO_EQ_H3PO4 = 98.0

EPS = 1e-9

TANQUES_FERTILIZANTES = {
    "NQ": "Tanque 1 — Nitratos y quelatos",
    "FOS": "Tanque 2 — Fosfatos",
    "SUL": "Tanque 3 — Sulfatos",
}
TANQUES_VALIDOS = list(TANQUES_FERTILIZANTES)


# =============================================================================
# 1. UTILIDADES NUMÉRICAS


def num(valor, default=0.0):
    """Convierte de forma segura a float, evitando None, NaN y errores."""
    try:
        if valor is None:
            return float(default)
        v = float(valor)
        if np.isnan(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def nnls_resolver(A, b, iters=4000):
    """Resuelve min ||Ax - b||² con la restricción x >= 0."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.size == 0 or b.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)

    try:
        from scipy.optimize import nnls as scipy_nnls
        x, _ = scipy_nnls(A, b)
        return x
    except Exception:
        pass

    # --- Fallback: descenso de gradiente proyectado (sin dependencias) ---
    n = A.shape[1]
    x = np.zeros(n)
    AtA = A.T @ A
    Atb = A.T @ b
    # Constante de Lipschitz (norma espectral) para tasa de aprendizaje segura
    try:
        L = np.linalg.norm(AtA, 2)
    except Exception:
        L = np.trace(AtA) if AtA.size else 1.0
    if not np.isfinite(L) or L <= EPS:
        L = 1.0
    lr = 1.0 / L
    for _ in range(iters):
        grad = AtA @ x - Atb
        x = x - lr * grad
        np.clip(x, 0, None, out=x)
    return x


def incompatibilidades_por_tanque(catalogo):
    """Devuelve conflictos de calcio con sulfatos o fosfatos por tanque."""
    conflictos = []
    for tanque in TANQUES_VALIDOS:
        sub = catalogo[catalogo["Tanque"] == tanque]
        tiene_ca = (sub["Ca"] > 0).any()
        tiene_s_o_p = ((sub["S"] > 0) | (sub["P2O5"] > 0)).any()
        if tiene_ca and tiene_s_o_p:
            nombres = sub[(sub["Ca"] > 0) | (sub["S"] > 0) | (sub["P2O5"] > 0)]["Fertilizante"].tolist()
            conflictos.append(
                f"⚠️ Incompatibilidad en {TANQUES_FERTILIZANTES[tanque]}: calcio junto a sulfatos o fosfatos "
                f"({', '.join(nombres)}). Reasigna los fertilizantes a tanques distintos."
            )
    return conflictos


def balance_iones_agua(agua):
    """Calcula cationes y aniones medidos en el agua, expresados en meq/L."""
    cationes = {
        "Na⁺": num(agua["Na"]) / 22.99,
        "K⁺": num(agua["K"]) / 39.10,
        "Ca²⁺": num(agua["Ca"]) * 2.0 / 40.08,
        "Mg²⁺": num(agua["Mg"]) * 2.0 / 24.31,
    }
    aniones = {
        "HCO₃⁻": num(agua["HCO3"]) / 61.0,
        "Cl⁻": num(agua["Cl"]) / 35.45,
        "NO₃⁻": num(agua["NO3"]) / 62.0,
        "SO₄²⁻": num(agua["SO4"]) * 2.0 / 96.06,
    }
    total_cationes = sum(cationes.values())
    total_aniones = sum(aniones.values())
    return {
        "cationes": cationes,
        "aniones": aniones,
        "total_cationes": total_cationes,
        "total_aniones": total_aniones,
        "diferencia": total_cationes - total_aniones,
    }


# =============================================================================
# 2. DATOS POR DEFECTO (PLANTILLAS DE CULTIVO Y CATÁLOGO DE FERTILIZANTES)
# =============================================================================

PLANTILLAS_CULTIVO = {
    "Tomate - Vegetativo": {"N": 150, "P": 50, "K": 150, "Ca": 150, "Mg": 40, "S": 60, "Fe": 3.0, "Zn": 0.3, "B": 0.3},
    "Tomate - Floración":  {"N": 180, "P": 60, "K": 250, "Ca": 180, "Mg": 50, "S": 70, "Fe": 3.0, "Zn": 0.3, "B": 0.3},
    "Tomate - Llenado de fruto": {"N": 200, "P": 60, "K": 300, "Ca": 200, "Mg": 60, "S": 80, "Fe": 3.0, "Zn": 0.3, "B": 0.3},
    "Pimiento - General":  {"N": 150, "P": 50, "K": 200, "Ca": 150, "Mg": 45, "S": 60, "Fe": 3.0, "Zn": 0.3, "B": 0.3},
    "Pepino - General":    {"N": 180, "P": 60, "K": 250, "Ca": 170, "Mg": 50, "S": 65, "Fe": 3.0, "Zn": 0.3, "B": 0.3},
    "Personalizado (definir manualmente)": {"N": 0, "P": 0, "K": 0, "Ca": 0, "Mg": 0, "S": 0, "Fe": 0.0, "Zn": 0.0, "B": 0.0},
}

# Catálogo de fertilizantes de los tres tanques de fertilizantes (composición % en peso)
CATALOGO_DEFECTO = pd.DataFrame([
    {"Fertilizante": "Nitrato de Calcio",           "Tanque": "NQ",  "N": 15.5, "P2O5": 0.0,  "K2O": 0.0,  "Ca": 19.0, "Mg": 0.0,  "S": 0.0,  "Fe": 0.0, "Zn": 0.0,  "B": 0.0},
    {"Fertilizante": "Nitrato de Potasio",          "Tanque": "NQ",  "N": 13.0, "P2O5": 0.0,  "K2O": 46.0, "Ca": 0.0,  "Mg": 0.0, "S": 0.0,  "Fe": 0.0, "Zn": 0.0,  "B": 0.0},
    {"Fertilizante": "Quelato de Hierro EDDHA",     "Tanque": "NQ",  "N": 0.0,  "P2O5": 0.0,  "K2O": 0.0,  "Ca": 0.0,  "Mg": 0.0, "S": 0.0, "Fe": 6.0, "Zn": 0.0, "B": 0.0},
    {"Fertilizante": "MAP (Fosfato Monoamónico)",   "Tanque": "FOS", "N": 12.0, "P2O5": 61.0, "K2O": 0.0,  "Ca": 0.0,  "Mg": 0.0, "S": 0.0, "Fe": 0.0, "Zn": 0.0, "B": 0.0},
    {"Fertilizante": "MKP (Fosfato Monopotásico)",  "Tanque": "FOS", "N": 0.0,  "P2O5": 52.0, "K2O": 34.0, "Ca": 0.0,  "Mg": 0.0, "S": 0.0, "Fe": 0.0, "Zn": 0.0, "B": 0.0},
    {"Fertilizante": "Sulfato de Magnesio",         "Tanque": "SUL", "N": 0.0,  "P2O5": 0.0,  "K2O": 0.0,  "Ca": 0.0,  "Mg": 9.8, "S": 13.0, "Fe": 0.0, "Zn": 0.0, "B": 0.0},
    {"Fertilizante": "Sulfato de Potasio",          "Tanque": "SUL", "N": 0.0,  "P2O5": 0.0,  "K2O": 50.0, "Ca": 0.0, "Mg": 0.0, "S": 18.0, "Fe": 0.0, "Zn": 0.0, "B": 0.0},
    {"Fertilizante": "Sulfato de Zinc",             "Tanque": "SUL", "N": 0.0,  "P2O5": 0.0,  "K2O": 0.0, "Ca": 0.0, "Mg": 0.0, "S": 0.0, "Fe": 0.0, "Zn": 35.0, "B": 0.0},
])

FERTILIZANTES_DISPONIBLES = CATALOGO_DEFECTO["Fertilizante"].tolist()

COLS_RIQUEZA = ["N", "P2O5", "K2O", "Ca", "Mg", "S", "Fe", "Zn", "B"]

# Ácidos por defecto para el Tanque 4
ACIDOS_DEFECTO = {
    "H3PO4": {"nombre": "Ácido Fosfórico", "concentracion": 85.0, "densidad": 1.685, "elemento": "P", "peso_eq": PESO_EQ_H3PO4, "ppm_por_meq": 31.0},
}


# =============================================================================
# 3. INICIALIZACIÓN DEL ESTADO DE SESIÓN
# =============================================================================

def init_state():
    defaults = {
        "agua": {
            "pH": 7.4, "CE": 0.35,
            "NO3": 5.0, "P": 0.2, "K": 3.0, "Ca": 40.0, "Mg": 12.0,
            "SO4": 15.0, "Na": 20.0, "Cl": 25.0, "HCO3": 180.0,
            "Fe": 0.05, "Zn": 0.02, "B": 0.05,
        },
        "plantilla_sel": "Tomate - Vegetativo",
        "meta": dict(PLANTILLAS_CULTIVO["Tomate - Vegetativo"]),
        "catalogo": CATALOGO_DEFECTO.copy(),
        "acidos_cfg": {k: dict(v) for k, v in ACIDOS_DEFECTO.items()},
        "hco3_residual": 0.5,       # meq/L deseados residuales
        "pH_objetivo": 5.8,
        "volumen_riego_m3": 10.0,
        "factor_concentracion": 100.0,
        "ce_objetivo": 2.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    catalogo = st.session_state["catalogo"].copy()
    if "Tanque" in catalogo.columns:
        catalogo = catalogo[catalogo["Fertilizante"] != "Solubor"].reset_index(drop=True)
        catalogo["Tanque"] = catalogo["Tanque"].replace({"A": "NQ", "B": "FOS", "NIT": "SUL"})
        catalogo.loc[catalogo["Fertilizante"] == "Nitrato de Potasio", "Tanque"] = "NQ"
        catalogo.loc[~catalogo["Tanque"].isin(TANQUES_VALIDOS), "Tanque"] = "SUL"
        st.session_state["catalogo"] = catalogo
    for codigo in TANQUES_VALIDOS:
        clave = f"fertilizantes_{codigo}"
        if clave not in st.session_state:
            st.session_state[clave] = st.session_state["catalogo"].loc[
                st.session_state["catalogo"]["Tanque"] == codigo, "Fertilizante"
            ].tolist()


init_state()


def aplicar_plantilla():
    nombre = st.session_state["plantilla_sel"]
    plantilla = PLANTILLAS_CULTIVO.get(nombre, {})
    for n in NUTRIENTES:
        st.session_state[f"meta_{n}"] = float(plantilla.get(n, 0.0))


# =============================================================================
# 4. INTERFAZ: ENCABEZADO Y PESTAÑAS
# =============================================================================

st.title("🌱 Calculadora de Soluciones Nutritivas para Fertirriego")
st.caption("Casa malla / ambiente protegido — cuatro ventanas de configuración")

tab_agua, tab_cultivo, tab_ferti, tab_tanques, tab_acidos, tab_ce, tab_resultados = st.tabs(
    ["💧 Agua de Riego", "🌱 Demanda del Cultivo", "🧪 Fertilizantes",
    "🛢️ Tanques Madre", "⚗️ Ácidos", "📈 CE objetivo", "📊 Resultados"]
)

# -----------------------------------------------------------------------
# MÓDULO A: AGUA DE RIEGO
# -----------------------------------------------------------------------
with tab_agua:
    st.subheader("Análisis físico-químico del agua de origen")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["agua"]["pH"] = st.number_input(
            "pH del agua", 0.0, 14.0, num(st.session_state["agua"]["pH"], 7.4), step=0.1)
        st.session_state["agua"]["CE"] = st.number_input(
            "Conductividad Eléctrica (dS/m)", 0.0, 10.0,
            num(st.session_state["agua"]["CE"], 0.35), step=0.01)
        st.session_state["agua"]["HCO3"] = st.number_input(
            "Bicarbonatos HCO3⁻ (ppm)", 0.0, 2000.0,
            num(st.session_state["agua"]["HCO3"], 180.0), step=1.0)
    with c2:
        st.session_state["agua"]["Na"] = st.number_input(
            "Sodio Na (ppm)", 0.0, 2000.0, num(st.session_state["agua"]["Na"], 20.0), step=1.0)
        st.session_state["agua"]["Cl"] = st.number_input(
            "Cloruros Cl⁻ (ppm)", 0.0, 2000.0, num(st.session_state["agua"]["Cl"], 25.0), step=1.0)

    st.markdown("---")
    st.subheader("Aportes nutricionales del agua (ppm)")
    st.caption("El Nitrato y el Sulfato se ingresan como ión y se convierten "
               "automáticamente a N y S elemental. El resto se ingresa en forma elemental.")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.session_state["agua"]["NO3"] = st.number_input(
            "Nitratos NO3⁻ (ppm)", 0.0, 1000.0, num(st.session_state["agua"]["NO3"], 5.0), step=0.5)
        st.session_state["agua"]["P"] = st.number_input(
            "Fósforo P (ppm)", 0.0, 200.0, num(st.session_state["agua"]["P"], 0.2), step=0.1)
        st.session_state["agua"]["K"] = st.number_input(
            "Potasio K (ppm)", 0.0, 500.0, num(st.session_state["agua"]["K"], 3.0), step=0.5)
    with c4:
        st.session_state["agua"]["Ca"] = st.number_input(
            "Calcio Ca (ppm)", 0.0, 500.0, num(st.session_state["agua"]["Ca"], 40.0), step=1.0)
        st.session_state["agua"]["Mg"] = st.number_input(
            "Magnesio Mg (ppm)", 0.0, 500.0, num(st.session_state["agua"]["Mg"], 12.0), step=0.5)
        st.session_state["agua"]["SO4"] = st.number_input(
            "Sulfatos SO4²⁻ (ppm)", 0.0, 1000.0, num(st.session_state["agua"]["SO4"], 15.0), step=1.0)
    with c5:
        st.session_state["agua"]["Fe"] = st.number_input(
            "Hierro Fe (ppm)", 0.0, 50.0, num(st.session_state["agua"]["Fe"], 0.05), step=0.01)
        st.session_state["agua"]["Zn"] = st.number_input(
            "Zinc Zn (ppm)", 0.0, 50.0, num(st.session_state["agua"]["Zn"], 0.02), step=0.01)
        st.session_state["agua"]["B"] = st.number_input(
            "Boro B (ppm)", 0.0, 50.0, num(st.session_state["agua"]["B"], 0.05), step=0.01)

# -----------------------------------------------------------------------
# MÓDULO B: DEMANDA NUTRICIONAL DEL CULTIVO
# -----------------------------------------------------------------------
with tab_cultivo:
    st.subheader("Plantilla de requerimiento (ppm meta)")
    st.selectbox(
        "Plantilla predeterminada",
        list(PLANTILLAS_CULTIVO.keys()),
        key="plantilla_sel",
        on_change=aplicar_plantilla,
    )
    if not any(f"meta_{n}" in st.session_state for n in NUTRIENTES):
        aplicar_plantilla()

    st.caption("Puedes sobrescribir manualmente cualquier valor meta antes de calcular.")
    cols = st.columns(len(NUTRIENTES))
    for i, n in enumerate(NUTRIENTES):
        with cols[i]:
            default_val = st.session_state.get(f"meta_{n}", PLANTILLAS_CULTIVO[st.session_state["plantilla_sel"]].get(n, 0.0))
            st.session_state[f"meta_{n}"] = st.number_input(
                f"{n} (ppm)", 0.0, 1000.0, num(default_val, 0.0),
                step=0.1 if n in ("Fe", "Zn", "B") else 1.0, key=f"input_meta_{n}")

    meta = {n: num(st.session_state.get(f"input_meta_{n}", st.session_state.get(f"meta_{n}", 0.0))) for n in NUTRIENTES}
    st.session_state["meta"] = meta

    # Demanda neta = meta - aporte de agua (informativo en esta pestaña)
    agua = st.session_state["agua"]
    aporte_agua = {
        "N": num(agua["NO3"]) * F_NO3_A_N,
        "P": num(agua["P"]),
        "K": num(agua["K"]),
        "Ca": num(agua["Ca"]),
        "Mg": num(agua["Mg"]),
        "S": num(agua["SO4"]) * F_SO4_A_S,
        "Fe": num(agua["Fe"]),
        "Zn": num(agua["Zn"]),
        "B": num(agua["B"]),
    }
    demanda_neta_preview = {n: max(meta[n] - aporte_agua[n], 0.0) for n in NUTRIENTES}

    st.markdown("---")
    st.subheader("Demanda neta (meta del cultivo − aporte del agua)")
    df_demanda = pd.DataFrame({
        "Nutriente": [NOMBRES_NUTRIENTES[n] for n in NUTRIENTES],
        "Meta (ppm)": [round(meta[n], 3) for n in NUTRIENTES],
        "Aporte Agua (ppm)": [round(aporte_agua[n], 3) for n in NUTRIENTES],
        "Demanda Neta (ppm)": [round(demanda_neta_preview[n], 3) for n in NUTRIENTES],
    })
    st.dataframe(df_demanda, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------
# MÓDULO C: BASE DE DATOS DE FERTILIZANTES
# -----------------------------------------------------------------------
with tab_ferti:
    st.subheader("Selección de fertilizantes por tanque")
    st.caption("Selecciona uno o varios fertilizantes por tanque. La composición se toma del catálogo de productos.")

    tablas_tanques = []
    for codigo, nombre in TANQUES_FERTILIZANTES.items():
        seleccionados = st.multiselect(
            nombre,
            options=FERTILIZANTES_DISPONIBLES,
            default=st.session_state[f"fertilizantes_{codigo}"],
            key=f"selector_fertilizantes_{codigo}",
        )
        st.session_state[f"fertilizantes_{codigo}"] = seleccionados
        filas = CATALOGO_DEFECTO[CATALOGO_DEFECTO["Fertilizante"].isin(seleccionados)].copy()
        filas["Tanque"] = codigo
        tablas_tanques.append(filas)
    st.session_state["catalogo"] = pd.concat(tablas_tanques, ignore_index=True)

# -----------------------------------------------------------------------
# MÓDULO D: TANQUES MADRE
# -----------------------------------------------------------------------
with tab_tanques:
    st.subheader("Configuración de Tanques Madre")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["volumen_riego_m3"] = st.number_input(
            "Volumen de la mezcla final de riego a preparar (m³)",
            0.01, 100000.0, num(st.session_state["volumen_riego_m3"], 10.0), step=1.0)
    with c2:
        st.session_state["factor_concentracion"] = st.number_input(
            "Factor de concentración del Tanque Madre (X)",
            1.0, 1000.0, num(st.session_state["factor_concentracion"], 100.0), step=1.0)

    st.info(
        f"Volumen físico de cada tanque concentrado ≈ "
        f"**{num(st.session_state['volumen_riego_m3']) / max(num(st.session_state['factor_concentracion']), EPS):.3f} m³** "
        f"({num(st.session_state['volumen_riego_m3']) * 1000 / max(num(st.session_state['factor_concentracion']), EPS):.1f} L), "
        f"inyectado al {100.0 / max(num(st.session_state['factor_concentracion']), EPS):.2f} % en el riego final."
    )

# -----------------------------------------------------------------------
# MÓDULO E: ÁCIDOS
# -----------------------------------------------------------------------
with tab_acidos:
    st.subheader("Tanque 4 — Ácidos")
    st.caption("Neutralización de bicarbonatos y acondicionamiento de pH.")
    c3, c4 = st.columns(2)
    with c3:
        st.session_state["hco3_residual"] = st.number_input(
            "Bicarbonatos residuales deseados (meq/L)",
            0.0, 10.0, num(st.session_state["hco3_residual"], 0.5), step=0.1)
    with c4:
        st.session_state["pH_objetivo"] = st.number_input(
            "pH objetivo después de acidificar", 3.0, 8.0,
            num(st.session_state["pH_objetivo"], 5.8), step=0.1)

    st.caption(
        "La dosis se calcula por neutralización de bicarbonatos y solo se activa si el pH objetivo es menor "
        "que el pH del agua. Se utiliza únicamente H₃PO₄, con primera equivalencia: "
        "98 g/mol por meq y 31 mg de P por meq."
    )

    st.caption("Parámetros comerciales de los ácidos (editables)")
    ac = st.session_state["acidos_cfg"]
    st.markdown("**Ácido Fosfórico (H₃PO₄)**")
    ac["H3PO4"]["concentracion"] = st.number_input(
        "Concentración comercial H₃PO₄ (% p/p)", 1.0, 100.0,
        num(ac["H3PO4"]["concentracion"], 85.0), step=1.0, key="h3po4_conc")
    ac["H3PO4"]["densidad"] = st.number_input(
        "Densidad H₃PO₄ (g/mL)", 0.5, 2.5, num(ac["H3PO4"]["densidad"], 1.685), step=0.001, key="h3po4_dens")
    st.session_state["acidos_cfg"] = ac

# -----------------------------------------------------------------------
# MÓDULO F: CONDUCTIVIDAD ELÉCTRICA
# -----------------------------------------------------------------------
with tab_ce:
    st.subheader("Conductividad Eléctrica objetivo")
    st.session_state["ce_objetivo"] = st.number_input(
        "CE deseada de la solución final (dS/m)",
        0.0, 10.0, num(st.session_state["ce_objetivo"], 2.0), step=0.01)
    st.caption(
        "La CE objetivo se usa como referencia y advertencia; la dosificación sigue priorizando "
        "las metas de nutrientes. Confirma la CE real con un conductímetro."
    )


# =============================================================================
# 5. MOTOR DE CÁLCULO (MÓDULO G)
# =============================================================================

def calcular_todo():
    agua = st.session_state["agua"]
    meta = st.session_state["meta"]
    catalogo = st.session_state["catalogo"].copy()
    acidos_cfg = st.session_state["acidos_cfg"]
    avisos = []
    balance_iones = balance_iones_agua(agua)

    # --- 1. Aporte del agua (elemental) ---
    aporte_agua = {
        "N": num(agua["NO3"]) * F_NO3_A_N,
        "P": num(agua["P"]),
        "K": num(agua["K"]),
        "Ca": num(agua["Ca"]),
        "Mg": num(agua["Mg"]),
        "S": num(agua["SO4"]) * F_SO4_A_S,
        "Fe": num(agua["Fe"]),
        "Zn": num(agua["Zn"]),
        "B": num(agua["B"]),
    }

    # --- 2. Demanda neta tras el agua ---
    demanda_tras_agua = {n: max(num(meta.get(n, 0.0)) - aporte_agua[n], 0.0) for n in NUTRIENTES}

    # --- 3. Neutralización de bicarbonatos (Tanque 4) ---
    hco3_ppm = num(agua["HCO3"])
    meq_hco3_agua = hco3_ppm / HCO3_EQ
    meq_residual = num(st.session_state["hco3_residual"])
    pH_agua = num(agua["pH"])
    pH_objetivo = num(st.session_state["pH_objetivo"], 5.8)
    requiere_bajar_pH = pH_objetivo < pH_agua - EPS
    meq_a_neutralizar = max(meq_hco3_agua - meq_residual, 0.0) if requiere_bajar_pH else 0.0

    meq_h3po4 = meq_a_neutralizar

    aporte_acido = {n: 0.0 for n in NUTRIENTES}
    aporte_acido["P"] = meq_h3po4 * ACIDOS_DEFECTO["H3PO4"]["ppm_por_meq"]
    for nutriente in ["P"]:
        if aporte_acido[nutriente] > demanda_tras_agua[nutriente] + EPS:
            avisos.append(
                f"El ácido aporta {aporte_acido[nutriente]:.2f} ppm de {nutriente}, "
                f"por encima de la demanda restante ({demanda_tras_agua[nutriente]:.2f} ppm). "
                "Reduce la proporción de ese ácido para evitar sobrepasar la meta."
            )

    # Volumen comercial de ácidos requerido (por m3 y total para el volumen de riego)
    v_riego = num(st.session_state["volumen_riego_m3"], 1.0)
    factor_c = max(num(st.session_state["factor_concentracion"], 100.0), EPS)

    def volumen_comercial_L_por_m3(meq_por_L, peso_eq, concentracion_pct, densidad):
        concentracion_pct = max(concentracion_pct, EPS)
        densidad = max(densidad, EPS)
        masa_pura_mg_L = meq_por_L * peso_eq           # mg de ácido puro por L
        masa_comercial_g_L = (masa_pura_mg_L / 1000.0) / (concentracion_pct / 100.0)
        volumen_mL_L = masa_comercial_g_L / densidad     # mL de producto comercial por L de mezcla
        volumen_L_por_m3 = volumen_mL_L * 1000.0 / 1000.0  # (mL/L)*(1000 L/m3)/(1000 mL/L) = L/m3
        return volumen_L_por_m3

    vol_h3po4_L_m3 = volumen_comercial_L_por_m3(
        meq_h3po4, PESO_EQ_H3PO4, acidos_cfg["H3PO4"]["concentracion"], acidos_cfg["H3PO4"]["densidad"])
    vol_h3po4_total_tanqueC = vol_h3po4_L_m3 * v_riego

    # --- 4. Demanda neta final para fertilizantes de los Tanques 1, 2 y 3 ---
    demanda_final = {n: max(demanda_tras_agua[n] - aporte_acido[n], 0.0) for n in NUTRIENTES}

    # --- 5. Preparar matriz de composición del catálogo (fracción elemental) ---
    catalogo = catalogo.fillna(0.0)
    for c in COLS_RIQUEZA:
        if c not in catalogo.columns:
            catalogo[c] = 0.0
        catalogo[c] = catalogo[c].apply(lambda v: num(v, 0.0))
    catalogo["Fertilizante"] = catalogo["Fertilizante"].fillna("").astype(str)
    catalogo = catalogo[catalogo["Fertilizante"].str.strip() != ""].reset_index(drop=True)
    if "Tanque" not in catalogo.columns:
        catalogo["Tanque"] = "SUL"
    catalogo["Tanque"] = catalogo["Tanque"].fillna("SUL").astype(str).str.upper().str.strip()
    catalogo["Tanque"] = catalogo["Tanque"].replace({"A": "NQ", "B": "FOS", "NIT": "SUL"})
    catalogo.loc[~catalogo["Tanque"].isin(TANQUES_VALIDOS), "Tanque"] = "SUL"
    catalogo.loc[catalogo["Fertilizante"] == "Nitrato de Potasio", "Tanque"] = "NQ"

    n_fert = len(catalogo)
    A_matrix = np.zeros((len(NUTRIENTES), max(n_fert, 1)))
    if n_fert > 0:
        for j, row in catalogo.iterrows():
            comp = {
                "N": row["N"] / 100.0,
                "P": (row["P2O5"] / 100.0) * F_P2O5_A_P,
                "K": (row["K2O"] / 100.0) * F_K2O_A_K,
                "Ca": row["Ca"] / 100.0,
                "Mg": row["Mg"] / 100.0,
                "S": row["S"] / 100.0,
                "Fe": row["Fe"] / 100.0,
                "Zn": row["Zn"] / 100.0,
                "B": row["B"] / 100.0,
            }
            for i, n in enumerate(NUTRIENTES):
                A_matrix[i, j] = comp[n]

    b_vector = np.array([demanda_final[n] for n in NUTRIENTES], dtype=float)

    if n_fert > 0:
        conflictos = incompatibilidades_por_tanque(catalogo)
        if conflictos:
            avisos.extend(conflictos)
            avisos.append("No se calcula la dosificación mientras exista una incompatibilidad dentro de un tanque.")
            dosis_g_m3 = np.zeros(n_fert)
            aporte_calculado_vector = np.zeros(len(NUTRIENTES))
        else:
            dosis_g_m3 = nnls_resolver(A_matrix, b_vector)
            dosis_g_m3 = np.clip(dosis_g_m3, 0, None)
            aporte_calculado_vector = A_matrix @ dosis_g_m3
    else:
        dosis_g_m3 = np.array([])
        aporte_calculado_vector = np.zeros(len(NUTRIENTES))
        avisos.append("No hay fertilizantes cargados en el catálogo: no se puede calcular la dosificación.")

    aporte_fertilizantes = {n: aporte_calculado_vector[i] for i, n in enumerate(NUTRIENTES)}

    catalogo["Dosis (g/m3)"] = dosis_g_m3 if n_fert > 0 else []
    catalogo["kg totales a disolver"] = catalogo["Dosis (g/m3)"] * v_riego / 1000.0 if n_fert > 0 else []

    # --- 6. Balance final ---
    total_aportado = {
        n: aporte_agua[n] + aporte_acido[n] + aporte_fertilizantes.get(n, 0.0)
        for n in NUTRIENTES
    }
    balance = {n: total_aportado[n] - num(meta.get(n, 0.0)) for n in NUTRIENTES}

    # --- 7. CE estimada (aproximación por masa total de productos) ---
    masa_fertilizantes_g_m3 = float(np.sum(dosis_g_m3)) if n_fert > 0 else 0.0
    masa_acidos_g_m3 = (
        vol_h3po4_L_m3 * acidos_cfg["H3PO4"]["densidad"] * 1000.0
    )
    ce_incremento = (masa_fertilizantes_g_m3 + masa_acidos_g_m3) / 640.0
    ce_estimada = num(agua["CE"]) + ce_incremento
    ce_objetivo = num(st.session_state["ce_objetivo"], 2.0)
    if ce_objetivo < num(agua["CE"]):
        avisos.append(
            f"La CE objetivo ({ce_objetivo:.2f} dS/m) es menor que la CE del agua de origen "
            f"({num(agua['CE']):.2f} dS/m); no puede alcanzarse agregando fertilizantes."
        )
    elif ce_estimada > ce_objetivo + 0.05:
        avisos.append(
            f"La CE estimada ({ce_estimada:.2f} dS/m) supera la CE objetivo "
            f"({ce_objetivo:.2f} dS/m). Reduce la concentración nutricional o el volumen de fertilizantes."
        )
    elif ce_estimada < ce_objetivo - 0.05:
        avisos.append(
            f"La CE estimada ({ce_estimada:.2f} dS/m) queda por debajo de la CE objetivo "
            f"({ce_objetivo:.2f} dS/m). La CE es orientativa y no se añadirá fertilizante solo para subirla."
        )

    # --- 8. Avisos de operación ---
    if not requiere_bajar_pH:
        avisos.append("El pH objetivo no es menor que el pH del agua; se desactivó la dosificación de ácidos.")
    if meq_a_neutralizar <= 0:
        avisos.append("El nivel de bicarbonatos del agua ya está por debajo (o igual) del residual deseado; "
                       "no se requiere neutralización adicional con ácidos.")

    return {
        "aporte_agua": aporte_agua,
        "demanda_tras_agua": demanda_tras_agua,
        "meq_hco3_agua": meq_hco3_agua,
        "meq_a_neutralizar": meq_a_neutralizar,
        "pH_agua": pH_agua,
        "pH_objetivo": pH_objetivo,
        "meq_h3po4": meq_h3po4,
        "aporte_acido": aporte_acido,
        "vol_h3po4_L_m3": vol_h3po4_L_m3,
        "vol_h3po4_total": vol_h3po4_total_tanqueC,
        "demanda_final": demanda_final,
        "catalogo_resultado": catalogo,
        "aporte_fertilizantes": aporte_fertilizantes,
        "total_aportado": total_aportado,
        "balance": balance,
        "ce_estimada": ce_estimada,
        "ce_objetivo": ce_objetivo,
        "avisos": avisos,
        "v_riego": v_riego,
        "factor_c": factor_c,
        "balance_iones": balance_iones,
    }


# =============================================================================
# 6. PESTAÑA DE RESULTADOS
# =============================================================================
with tab_resultados:
    st.subheader("Cálculo de la solución nutritiva")
    calcular = st.button("🧮 Calcular Solución Nutritiva", type="primary", use_container_width=True)

    if calcular or "ultimo_resultado" in st.session_state:
        try:
            if calcular:
                resultado = calcular_todo()
                st.session_state["ultimo_resultado"] = resultado
            else:
                resultado = st.session_state["ultimo_resultado"]
        except Exception as e:
            st.error(f"Ocurrió un error al calcular la solución nutritiva: {e}")
            st.stop()

        for aviso in resultado["avisos"]:
            st.warning(aviso)

        # ---- Panel 1: Resumen de ajuste nutricional ----
        st.markdown("### 1️⃣ Resumen de Ajuste Nutricional (ppm)")
        meta = st.session_state["meta"]
        df_resumen = pd.DataFrame({
            "Nutriente": [NOMBRES_NUTRIENTES[n] for n in NUTRIENTES],
            "Requerido": [round(num(meta.get(n, 0.0)), 2) for n in NUTRIENTES],
            "Aportado por Agua": [round(resultado["aporte_agua"][n], 2) for n in NUTRIENTES],
            "Aportado por Ácidos (Tanque 4)": [round(resultado["aporte_acido"][n], 2) for n in NUTRIENTES],
            "Dosis Fertilizantes (Tanques 1, 2 y 3)": [round(resultado["aporte_fertilizantes"].get(n, 0.0), 2) for n in NUTRIENTES],
            "Total Aportado": [round(resultado["total_aportado"][n], 2) for n in NUTRIENTES],
            "Balance Final (ppm)": [round(resultado["balance"][n], 2) for n in NUTRIENTES],
        })
        st.dataframe(df_resumen, hide_index=True, use_container_width=True)

        st.markdown("---")
        # ---- Panel 2: Dosis de Tanques Madre ----
        st.markdown("### 2️⃣ Dosis de Tanques Madre")
        cat_res = resultado["catalogo_resultado"]
        for codigo, nombre in TANQUES_FERTILIZANTES.items():
            st.markdown(f"**{nombre}**")
            tanque_resultado = cat_res[cat_res["Tanque"] == codigo][
                ["Fertilizante", "Dosis (g/m3)", "kg totales a disolver"]
            ]
            if len(tanque_resultado):
                tanque_disp = tanque_resultado.copy()
                tanque_disp["Dosis (g/m3)"] = tanque_disp["Dosis (g/m3)"].round(2)
                tanque_disp["kg totales a disolver"] = tanque_disp["kg totales a disolver"].round(3)
                st.dataframe(tanque_disp, hide_index=True, use_container_width=True)
            else:
                st.caption(f"Sin fertilizantes asignados a {nombre}.")

        st.markdown("**Tanque 4 — Ácidos** (Acondicionamiento de pH)")
        df_tanqueC = pd.DataFrame({
            "Ácido": ["Ácido Fosfórico (H₃PO₄)"],
            "L comercial / m³ de riego": [round(resultado["vol_h3po4_L_m3"], 4)],
            f"L totales para {resultado['v_riego']:.1f} m³": [round(resultado["vol_h3po4_total"], 3)],
        })
        st.dataframe(df_tanqueC, hide_index=True, use_container_width=True)
        st.caption(
            f"meq/L de HCO₃⁻ del agua: {resultado['meq_hco3_agua']:.3f} · "
            f"meq/L a neutralizar: {resultado['meq_a_neutralizar']:.3f} "
            f"(H₃PO₄: {resultado['meq_h3po4']:.3f} meq/L)"
        )

        st.markdown("### 3️⃣ Balance de iones del agua de riego (meq/L)")
        balance_iones = resultado["balance_iones"]
        df_iones = pd.DataFrame({
            "Grupo": ["Cationes", "Aniones", "Diferencia (cationes − aniones)"],
            "Total (meq/L)": [
                round(balance_iones["total_cationes"], 3),
                round(balance_iones["total_aniones"], 3),
                round(balance_iones["diferencia"], 3),
            ],
        })
        st.dataframe(df_iones, hide_index=True, use_container_width=True)
        st.caption(
            "Cationes: Na⁺, K⁺, Ca²⁺ y Mg²⁺. Aniones: HCO₃⁻, Cl⁻, NO₃⁻ y SO₄²⁻. "
            "Un balance cercano a cero indica consistencia aproximada del análisis del agua."
        )

        vol_tanque_fisico_L = (resultado["v_riego"] * 1000.0) / resultado["factor_c"]
        st.info(f"Volumen físico sugerido para cada tanque concentrado: **{vol_tanque_fisico_L:.1f} L** "
                f"(factor {resultado['factor_c']:.0f}x sobre {resultado['v_riego']:.1f} m³ de riego final).")

        st.markdown("---")
        # ---- Panel 3: Indicadores de salida ----
        st.markdown("### 4️⃣ Indicadores de Salida")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("CE estimada de la solución final", f"{resultado['ce_estimada']:.2f} dS/m",
                       delta=f"{resultado['ce_estimada'] - resultado['ce_objetivo']:.2f} vs. objetivo")
        with c2:
            st.metric("CE deseada", f"{resultado['ce_objetivo']:.2f} dS/m")

    else:
        st.info("Configura las pestañas de Agua, Cultivo, Fertilizantes y Tanques, y presiona "
                "**Calcular Solución Nutritiva** para ver los resultados.")

st.markdown("---")
st.caption(
    "⚠️ Herramienta de apoyo técnico. Los valores de riqueza nutricional, densidades y concentraciones "
    "comerciales deben verificarse siempre contra la etiqueta del producto y las recomendaciones de un "
    "ingeniero agrónomo antes de su aplicación en campo."
)
