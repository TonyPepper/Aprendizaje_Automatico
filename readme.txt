pytoncrear entorno virtual
en la ubicacion donde desee hacer el medio abrir consola:

----por primera vez----
c: python -m venv entorno
c: entorno\Scripts\activate
(entorno) c:pip install -r .\requirements.txt
(entorno) c:streamlit run app_clima.py

---- segunda en adelante----
c: entorno\Scripts\activate
(entorno) c:streamlit run app_clima.py