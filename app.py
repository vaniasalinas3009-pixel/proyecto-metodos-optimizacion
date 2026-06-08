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
- Búsqueda de línea con condiciones de Wolfe

La aplicación permite ejecutar una gran cantidad de iteraciones. Si el método no logra converger,
se informa una explicación matemática del motivo.
""")

st.info("Curso: Métodos de Optimización | Primer Semestre 2026")

st.subheader("👥 Integrantes del grupo")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Vania Salinas Barra**")

with col2:
    st.write("**Vicente Barra Moraga**")

with col3:
    st.write("**Isidora Cárdenas Ferrada**")

st.divider()

st.subheader("⚙️ Parámetros de entrada")

izq, der = st.columns(2)

with izq:
    n_variables = st.number_input(
        "Número de variables",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    funcion_texto = st.text_input(
        "Función objetivo",
        "x1**2+x2**2"
    )

    metodo = st.selectbox(
        "Método de optimización",
        ["Gradiente", "Gradiente Conjugado", "Newton"]
    )

    punto_texto = st.text_input(
        "Punto inicial separado por comas",
        "2,3"
    )

with der:
    iteraciones_max = st.number_input(
        "Máximo de iteraciones",
        min_value=1,
        max_value=1000000,
        value=100,
        step=100
    )

    tolerancia = st.number_input(
        "Tolerancia",
        value=0.0001,
        format="%.8f"
    )

    wolfe_c1 = st.number_input(
        "Parámetro Wolfe c1",
        value=0.0001,
        format="%.8f"
    )

    wolfe_c2 = st.number_input(
        "Parámetro Wolfe c2",
        value=0.9,
        format="%.4f"
    )

st.caption(
    "Nota: computacionalmente no existen iteraciones infinitas. "
    "La aplicación permite un máximo alto de iteraciones y detiene el proceso "
    "si converge, si se estanca o si se detecta una imposibilidad numérica/matemática."
)

st.divider()


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


def obtener_alpha_wolfe(f, grad, x, direccion, c1, c2):
    try:
        resultado = line_search(
            f,
            grad,
            x,
            direccion,
            c1=c1,
            c2=c2
        )

        alpha = resultado[0]

        if alpha is None or alpha <= 0:
            alpha = 1.0

    except Exception:
        alpha = 1.0

    fx = f(x)
    gx = grad(x)

    intentos = 0

    while True:
        try:
            nuevo_valor = f(x + alpha * direccion)

            condicion_armijo = nuevo_valor <= fx + c1 * alpha * np.dot(gx, direccion)

            if condicion_armijo:
                break

            alpha *= 0.5
            intentos += 1

            if alpha < 1e-12 or intentos > 60:
                alpha = 0.01
                break

        except Exception:
            alpha *= 0.5
            intentos += 1

            if alpha < 1e-12 or intentos > 60:
                alpha = 0.01
                break

    return alpha


def diagnosticar_no_convergencia(metodo, error_final, max_iter, tolerancia):
    return (
        f"El método {metodo} no alcanzó la tolerancia solicitada dentro de las "
        f"{max_iter} iteraciones definidas. Matemáticamente, esto puede ocurrir porque "
        f"la función presenta una topología compleja, curvatura muy pronunciada, valles estrechos, "
        f"puntos de silla, ausencia de mínimo global alcanzable desde el punto inicial o porque "
        f"el método seleccionado converge lentamente para esta función. "
        f"El error final obtenido fue {error_final:.6e}, mientras que la tolerancia exigida fue "
        f"{tolerancia:.6e}. Por lo tanto, la aplicación no declara este resultado como mínimo exacto, "
        f"sino como el mejor punto aproximado alcanzado bajo las condiciones ingresadas."
    )


def optimizar(f, grad, hess, x0, metodo, max_iter, tolerancia, c1, c2):
    x = x0.astype(float)

    errores = []
    historial = []

    direccion = None
    grad_anterior = None

    convergio = False
    motivo_detencion = ""

    for k in range(int(max_iter)):
        try:
            valor = f(x)
            g = grad(x)
            error = np.linalg.norm(g)

            if not np.isfinite(valor) or not np.isfinite(error):
                motivo_detencion = (
                    "El proceso se detuvo porque se generaron valores no finitos "
                    "(NaN o infinito). Desde el punto de vista matemático, esto puede indicar "
                    "que la función no está definida en la región explorada, que existe una "
                    "singularidad, una división por cero, crecimiento no acotado o que el paso "
                    "del algoritmo llevó a una zona inválida del dominio."
                )
                break

            errores.append(error)

            historial.append({
                "Iteración": k + 1,
                "Valor función": valor,
                "Error": error,
                "Punto": np.round(x, 6)
            })

            if error < tolerancia:
                convergio = True
                motivo_detencion = (
                    "El método convergió correctamente. La norma del gradiente alcanzó un valor "
                    "menor que la tolerancia definida, por lo que se cumple el criterio de parada "
                    "matemático asociado a un punto estacionario."
                )
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
                    motivo_detencion = (
                        "Durante el método de Newton, la matriz Hessiana no fue invertible. "
                        "Esto puede ocurrir cuando la función no presenta curvatura suficiente "
                        "o cuando se evalúa cerca de un punto singular. Para evitar la detención "
                        "del algoritmo, se utilizó una dirección de descenso basada en el gradiente."
                    )

                if np.dot(direccion, g) >= 0:
                    direccion = -g

            alpha = obtener_alpha_wolfe(f, grad, x, direccion, c1, c2)

            x_nuevo = x + alpha * direccion

            if not np.all(np.isfinite(x_nuevo)):
                motivo_detencion = (
                    "El proceso se detuvo porque el nuevo punto generado contiene valores "
                    "no finitos. Esto indica inestabilidad numérica o avance hacia una región "
                    "matemáticamente inválida."
                )
                break

            if np.linalg.norm(x_nuevo - x) < 1e-12 and error >= tolerancia:
                motivo_detencion = (
                    "El proceso se detuvo porque el avance entre iteraciones fue prácticamente "
                    "nulo, pero no se alcanzó la tolerancia solicitada. Matemáticamente, esto "
                    "indica estancamiento numérico: el algoritmo no logra mejorar la solución "
                    "con el tamaño de paso disponible."
                )
                break

            grad_anterior = g.copy()
            x = x_nuevo

        except Exception as e:
            motivo_detencion = (
                "El proceso se detuvo por un error matemático o numérico durante la optimización: "
                + str(e)
            )
            break

    if len(errores) == 0:
        errores = [np.inf]
        historial.append({
            "Iteración": 0,
            "Valor función": np.nan,
            "Error": np.inf,
            "Punto": np.round(x, 6)
        })

    if not convergio and motivo_detencion == "":
        motivo_detencion = diagnosticar_no_convergencia(
            metodo,
            errores[-1],
            max_iter,
            tolerancia
        )

    try:
        valor_final = f(x)
    except Exception:
        valor_final = np.nan

    return (
        x,
        valor_final,
        errores[-1],
        len(errores),
        errores,
        historial,
        convergio,
        motivo_detencion
    )


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
    imagen_grafico,
    convergio,
    motivo_detencion
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
        "<b>Integrantes:</b><br/>"
        "Vania Salinas Barra<br/>"
        "Vicente Barra Moraga<br/>"
        "Isidora Cárdenas Ferrada",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.8*cm))

    contenido.append(
        Paragraph(
            """
            <b>Curso:</b> Métodos de Optimización<br/>
            <b>Fecha:</b> 08/06/2026<br/>
            <b>Profesor:</b> Gerardo Silva
            """,
            estilos["Normal"]
        )
    )

    contenido.append(PageBreak())

    contenido.append(Paragraph("<b>1. Resumen del proyecto</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "El presente informe resume los resultados obtenidos mediante una aplicación web "
        "desarrollada para resolver problemas de optimización numérica. La herramienta permite "
        "ingresar una función objetivo, seleccionar un método de optimización, definir un punto "
        "inicial, establecer una tolerancia y analizar el comportamiento de convergencia.",
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

    estado = "Convergió" if convergio else "No convergió bajo las condiciones definidas"

    datos_resultados = [
        ["Resultado", "Valor"],
        ["Estado del proceso", estado],
        ["Punto obtenido", str(np.round(punto_minimo, 6))],
        ["Valor de la función", str(round(valor_minimo, 6))],
        ["Iteraciones realizadas", str(iter_realizadas)],
        ["Error final", str(round(error_final, 8))]
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

    contenido.append(Paragraph("<b>4. Explicación matemática del criterio de parada</b>", estilos["Heading1"]))
    contenido.append(Paragraph(motivo_detencion, estilos["Normal"]))
    contenido.append(Spacer(1, 0.7*cm))

    contenido.append(Paragraph("<b>5. Análisis de convergencia</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "El gráfico de convergencia muestra la evolución del error medido como la norma del "
        "gradiente. Una disminución progresiva del error indica que el método avanza hacia un "
        "punto estacionario de la función objetivo. Si el error no disminuye lo suficiente, se "
        "concluye que el método no logró satisfacer la tolerancia definida.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.5*cm))

    grafico_pdf = Image(imagen_grafico, width=15*cm, height=8*cm)
    contenido.append(grafico_pdf)

    contenido.append(PageBreak())

    contenido.append(Paragraph("<b>6. Tabla de iteraciones</b>", estilos["Heading1"]))

    tabla_muestra = tabla.head(20).copy()
    datos_iteraciones = [["Iteración", "Valor función", "Error"]]

    for _, fila in tabla_muestra.iterrows():
        datos_iteraciones.append([
            int(fila["Iteración"]),
            round(float(fila["Valor función"]), 6),
            round(float(fila["Error"]), 8)
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

    contenido.append(Paragraph("<b>7. Conclusiones</b>", estilos["Heading1"]))

    contenido.append(Paragraph(
        "Durante el desarrollo del proyecto se logró implementar una aplicación web capaz de "
        "resolver problemas de optimización mediante los métodos de Gradiente, Gradiente Conjugado "
        "y Newton. Además, se incorporó una búsqueda de paso basada en condiciones de Wolfe para "
        "mejorar la estabilidad del proceso iterativo.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.3*cm))

    contenido.append(Paragraph(
        "Cuando el algoritmo no alcanza la tolerancia solicitada, la aplicación informa el motivo "
        "del criterio de parada, evitando declarar como óptima una solución que no cumple "
        "matemáticamente con la condición de convergencia establecida.",
        estilos["Normal"]
    ))

    contenido.append(Spacer(1, 0.3*cm))

    contenido.append(Paragraph(
        "El uso de gráficos, tablas e informes automáticos permite visualizar el comportamiento "
        "del error, analizar el desempeño de cada algoritmo y fundamentar técnicamente los "
        "resultados obtenidos.",
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

        (
            punto_minimo,
            valor_minimo,
            error_final,
            iter_realizadas,
            errores,
            historial,
            convergio,
            motivo_detencion
        ) = optimizar(
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

        if convergio:
            st.success(motivo_detencion)
        else:
            st.warning(motivo_detencion)

        st.subheader("📌 Resultados principales")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Método", metodo)
        r2.metric("Iteraciones", iter_realizadas)
        r3.metric("Valor función", round(valor_minimo, 6))
        r4.metric("Error final", round(error_final, 8))

        st.write("**Punto obtenido:**", np.round(punto_minimo, 6))

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
            imagen_grafico,
            convergio,
            motivo_detencion
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