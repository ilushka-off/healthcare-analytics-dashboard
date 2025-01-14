from dash import Dash
from layouts.dashboard import layout

# Инициализация приложения
app = Dash(__name__)
app.title = "Healthcare Analytics Dashboard"
app.layout = layout

def main():
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
