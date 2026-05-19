import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import io
import os

from scipy.optimize import line_search
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm


st.set_page_config(page_title="Métodos de Optimización", page_icon="📈", layout="wide")

st.title("📈 Proyecto Final: Métodos de Optimización")

st.markdown("""
Aplicación web para encontrar el mínimo de una función mediante:

- Método del Gradiente
- Método del Gradiente Conjugado
- Método de Newton
- Condiciones de Wolfe
""")

st.info("Curso: Métodos de Optimización | Primer Semestre 2026")

st.subheader("👥 Integrantes del grupo")

c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Vania Salinas**")

with c2:
    st.write("**Vicente Barra**")

with c3:
    st.write("**Isidora**")

st.divider()

st.subheader("⚙️ Parámetros de entrada")

izq, der = st.columns(2)

with izq:
    n_variables = st.number_input("Número de variables", min_value=1, max_value=5, value=2)
    funcion_texto = st.text_input("Función objetivo", "x1**2+x2**2")

    metodo = st.selectbox(
        "Método de optimización",
        ["Gradiente", "Gradiente Conjugado", "Newton"]
    )

    punto_texto = st.text_input("Punto inicial", "2,3")

with der:
    iteraciones_max = st.number_input("Máximo de iteraciones", min_value=1, value=100)
    tolerancia = st.number_input("Tolerancia", value=0.0001, format="%.6f")
    wolfe_c1 = st.number_input("Parámetro Wolfe c1", value=0.0001, format="%.6f")
    wolfe_c2 = st.number_input("Parámetro Wolfe c2", value=0.9, format="%.2f")


def preparar_funciones(funcion_texto, n):
    variables = sp.symbols(" ".join([f"x{i+1}" for i in range(n)]))

    if n == 1:
        variables = (variables,)

    funcion = sp.sympify(funcion_texto)

    gradiente = [sp.diff(funcion, v) for v in variables]

    hessiana = sp.Matrix([
        [sp.diff(g, v) for v in variables]
        for g in gradiente
    ])

    f_num = sp.lambdify(variables, funcion, "numpy")
    grad_num = sp.lambdify(variables, gradiente, "numpy")
    hess_num = sp.lambdify(variables, hessiana, "numpy")

    def f(x):
        return float(f_num(*x))

    def grad(x):
        return np.array(grad_num(*x), dtype=float).reshape(-1)

    def hess(x):
        return np.array(hess_num(*x), dtype=float)

    return f, grad, hess


def obtener_alpha(f, grad, x, direccion, c1, c2):
    try:
        alpha = line_search(f, grad, x, direccion, c1=c1, c2=c2)[0]

        if alpha is None or alpha <= 0:
            alpha = 1.0

    except Exception:
        alpha = 1.0

    fx = f(x)
    gx = grad(x)

    while f(x + alpha * direccion) > fx + c1 * alpha * np.dot(gx, direccion):
        alpha *= 0.5

        if alpha < 1e-8:
            alpha = 0.01
            break

    return alpha


def optimizar(f, grad, hess, x0, metodo, max_iter, tol, c1, c2):
    x = x0.astype(float)

    errores = []
    historial = []

    direccion = None
    grad_anterior = None

    for k in range(int(max_iter)):
        g = grad(x)
        error = np.linalg.norm(g)
        valor = f(x)

        errores.append(error)

        historial.append({
            "Iteración": k + 1,
            "Valor función": valor,
            "Error": error,
            "Punto": np.round(x, 6)
        })

        if error < tol:
            break

        if metodo == "Gradiente":
            direccion = -g

        elif metodo == "Gradiente Conjugado":
            if k == 0 or grad_anterior is None:
                direccion = -g
            else:
                beta = np.dot(g, g) / max(np.dot(grad_anterior, grad_anterior), 1e-12)
                direccion = -g + beta * direccion

            if np.dot(direccion, g) >= 0:
                direccion = -g

        elif metodo == "Newton":
            H = hess(x)

            try:
                direccion = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                direccion = -g

            if np.dot(direccion, g) >= 0:
                direccion = -g

        alpha = obtener_alpha(f, grad, x, direccion, c1, c2)

        grad_anterior = g.copy()
        x = x + alpha * direccion

    return x, f(x), errores[-1], len(errores), errores, historial


def generar_pdf(
    funcion_texto,
    metodo,
    punto_texto,
    iteraciones_max,
    tolerancia,
    wolfe_c1,
    wolfe_c2,
    punto_minimo,
    valor_minimo,
    error_final,
    iter_realizadas,
    tabla,
    imagen_grafico
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=1.8*cm,
        bottomMargin=1.8*cm
    )

    estilos = getSampleStyleSheet()
    contenido = []

    if os.path.exists("logo_universidad.png"):
        logo = Image("logo_universidad.png", width=7*cm, height=2.5*cm)
        contenido.append(logo)
        contenido.append(Spacer(1, 1*cm))

    contenido.append(Paragraph("<b>PROYECTO FINAL</b>", estilos["Title"]))
    contenido.append(Paragraph("<b>Métodos de Optimización</b>", estilos["Title"]))
    contenido.append(Spacer(1, 1*cm))

    contenido.append(Paragraph(
        "Aplicación web para la minimización de funciones mediante métodos numéricos.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 1*cm))

    contenido.append(Paragraph(
        "<b>Integrantes:</b><br/>Vania Salinas<br/>Vicente Barra<br/>Isidora",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.8*cm))

    contenido.append(Paragraph(
        "<b>Curso:</b> Métodos de Optimización<br/><b>Periodo:</b> Primer Semestre 2026",
        estilos["Normal"]
    ))

    contenido.append(PageBreak())

    contenido.append(Paragraph("<b>1. Resumen del proyecto</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "El presente informe resume los resultados obtenidos mediante una aplicación web "
        "desarrollada para resolver problemas de optimización numérica. La herramienta permite "
        "ingresar una función objetivo, seleccionar un método, definir un punto inicial y analizar "
        "el comportamiento de convergencia.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.6*cm))

    contenido.append(Paragraph("<b>2. Parámetros ingresados</b>", estilos["Heading1"]))

    datos_parametros = [
        ["Parámetro", "Valor"],
        ["Función objetivo", funcion_texto],
        ["Método seleccionado", metodo],
        ["Punto inicial", punto_texto],
        ["Máximo de iteraciones", str(iteraciones_max)],
        ["Tolerancia", str(tolerancia)],
        ["Wolfe c1", str(wolfe_c1)],
        ["Wolfe c2", str(wolfe_c2)]
    ]

    tabla_parametros = Table(datos_parametros, colWidths=[6*cm, 9*cm])
    tabla_parametros.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    contenido.append(tabla_parametros)
    contenido.append(Spacer(1, 0.7*cm))

    contenido.append(Paragraph("<b>3. Resultados obtenidos</b>", estilos["Heading1"]))

    datos_resultados = [
        ["Resultado", "Valor"],
        ["Punto mínimo encontrado", str(np.round(punto_minimo, 6))],
        ["Valor mínimo de la función", str(round(valor_minimo, 6))],
        ["Iteraciones realizadas", str(iter_realizadas)],
        ["Error final", str(round(error_final, 6))]
    ]

    tabla_resultados = Table(datos_resultados, colWidths=[6*cm, 9*cm])
    tabla_resultados.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#006699")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    contenido.append(tabla_resultados)
    contenido.append(Spacer(1, 0.7*cm))

    contenido.append(Paragraph("<b>4. Análisis de convergencia</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "El gráfico de convergencia muestra la evolución del error medido como la norma del "
        "gradiente. Una disminución progresiva del error indica que el método avanza hacia un "
        "punto estacionario de la función objetivo.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.5*cm))

    grafico_pdf = Image(imagen_grafico, width=15*cm, height=8*cm)
    contenido.append(grafico_pdf)

    contenido.append(PageBreak())

    contenido.append(Paragraph("<b>5. Tabla de iteraciones</b>", estilos["Heading1"]))

    tabla_muestra = tabla.head(20).copy()

    datos_iteraciones = [["Iteración", "Valor función", "Error"]]

    for _, fila in tabla_muestra.iterrows():
        datos_iteraciones.append([
            int(fila["Iteración"]),
            round(float(fila["Valor función"]), 6),
            round(float(fila["Error"]), 6)
        ])

    tabla_iter = Table(datos_iteraciones, colWidths=[4*cm, 5*cm, 5*cm])
    tabla_iter.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    contenido.append(tabla_iter)
    contenido.append(Spacer(1, 0.8*cm))

    contenido.append(Paragraph("<b>6. Conclusiones</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "Durante el desarrollo del proyecto se logró implementar una aplicación web capaz de "
        "resolver problemas de optimización mediante los métodos de Gradiente, Gradiente Conjugado "
        "y Newton. Además, se incorporó una búsqueda de paso basada en condiciones de Wolfe para "
        "mejorar la estabilidad de convergencia.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.3*cm))

    contenido.append(Paragraph(
        "Al comparar los métodos implementados se observó que Newton requiere menos iteraciones "
        "para alcanzar la solución óptima en funciones cuadráticas, mientras que el método del "
        "Gradiente presenta una convergencia más progresiva.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.3*cm))

    contenido.append(Paragraph(
        "El uso de gráficos y tablas permitió visualizar el comportamiento del error y analizar "
        "el desempeño de cada algoritmo durante el proceso iterativo.",
        estilos["Normal"]
    ))

    doc.build(contenido)
    buffer.seek(0)

    return buffer


if st.button("🚀 Ejecutar optimización"):

    try:
        punto_inicial = np.array([float(x.strip()) for x in punto_texto.split(",")])

        if len(punto_inicial) != int(n_variables):
            st.error("El punto inicial debe tener la misma cantidad de valores que el número de variables.")
            st.stop()

        f, grad, hess = preparar_funciones(funcion_texto, int(n_variables))

        punto_minimo, valor_minimo, error_final, iter_realizadas, errores, historial = optimizar(
            f,
            grad,
            hess,
            punto_inicial,
            metodo,
            int(iteraciones_max),
            tolerancia,
            wolfe_c1,
            wolfe_c2
        )

        st.success("Optimización ejecutada correctamente")

        st.subheader("📌 Resultados principales")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Método", metodo)
        r2.metric("Iteraciones", iter_realizadas)
        r3.metric("Valor mínimo", round(valor_minimo, 6))
        r4.metric("Error final", round(error_final, 6))

        st.write("**Punto mínimo encontrado:**", np.round(punto_minimo, 6))

        st.subheader("📉 Gráfico de convergencia")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, iter_realizadas + 1), errores, marker="o")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Error ||∇f(x)||")
        ax.set_title("Convergencia del método")
        ax.grid(True)

        st.pyplot(fig)

        imagen_grafico = io.BytesIO()
        fig.savefig(imagen_grafico, format="png", bbox_inches="tight")
        imagen_grafico.seek(0)

        st.subheader("📋 Tabla de iteraciones")

        tabla = pd.DataFrame(historial)

        st.dataframe(tabla)

        pdf = generar_pdf(
            funcion_texto,
            metodo,
            punto_texto,
            iteraciones_max,
            tolerancia,
            wolfe_c1,
            wolfe_c2,
            punto_minimo,
            valor_minimo,
            error_final,
            iter_realizadas,
            tabla,
            imagen_grafico
        )

        st.download_button(
            label="📄 Descargar informe PDF profesional",
            data=pdf,
            file_name="Informe_Optimizacion.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error("Error en los datos ingresados o en el proceso de optimización.")
        st.write(e)