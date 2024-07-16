import streamlit as st
import sys
import subprocess

st.title("BeautifulSoup Installation Check")

st.write("Python version:", sys.version)
st.write("Python executable:", sys.executable)

try:
    from bs4 import BeautifulSoup
    st.success("BeautifulSoup imported successfully!")
    st.write("BeautifulSoup version:", BeautifulSoup.__version__)
except ImportError as e:
    st.error(f"Failed to import BeautifulSoup: {str(e)}")

st.write("Installed packages:")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    st.code(result.stdout)
except Exception as e:
    st.error(f"Failed to list packages: {str(e)}")
