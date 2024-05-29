from flask import Flask, redirect, render_template, request, url_for
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Variables globales temporales para almacenar los datos entre las rutas y usarlas en el grafo
temp_data = {
    'costos': None,
    'ofertas': None,
    'demandas': None,
    'resultado': None
}

@app.route('/', methods=['GET', 'POST'])
def index(): 
    if request.method == 'POST' and request.form['action'] == 'solver':
        n = int(request.form['rows'])
        m = int(request.form['cols'])
        return render_template('table.html', rows=n, cols=m)
    elif request.method == 'POST' and request.form['action'] == 'costominimo':
        n = int(request.form['rows'])
        m = int(request.form['cols'])
        return render_template('costominimo.html', rows=n, cols=m)
    return render_template('index.html')
    
@app.route('/solver', methods=['POST'])
def solver():
    rows = int(request.form['N'])
    cols = int(request.form['M'])
    
    # Obtener los valores de las celdas de la tabla de costos
    costs = {}
    for i in range(rows):
        for j in range(cols):
            cell_value = float(request.form.get(f'cell_{i}_{j}', 0))
            costs[(i, j)] = cell_value

    # Obtener los valores de offers y demand
    offers = {}
    demand = {}

    for i in range(rows):
        offer_value = float(request.form.get(f'offer_{i}', 0))  
        offers[i] = offer_value
    
    for j in range(cols):
        demand_value = float(request.form.get(f'demand_{j}', 0))
        demand[j] = demand_value
    
    # Crear el problema de optimización
    problem = pulp.LpProblem("Problema_de_transporte", pulp.LpMinimize)

    # Variables de decisión
    sent_quantity = pulp.LpVariable.dicts(
        "Cantidad_enviada", 
        (offers.keys(), demand.keys()), 
        lowBound=0, 
        cat='Integer'
    )
        
    # Función objetivo: Minimizar el costo total de envío
    problem += pulp.lpSum([sent_quantity[i][j] * costs[(i, j)] for i in offers.keys() for j in demand.keys()])

    # Restricciones de oferta
    for i in offers.keys():
        problem += pulp.lpSum([sent_quantity[i][j] for j in demand.keys()]) <= offers[i], f"offer_{i}"

    # Restricciones de demanda
    for j in demand.keys():
        problem += pulp.lpSum([sent_quantity[i][j] for i in offers.keys()]) >= demand[j], f"demand_{j}"

    # Resolver el problema
    problem.solve()

    total = pulp.value(problem.objective)

    # Imprimir el estado del problema y las soluciones para depuración
    print(f"Status: {pulp.LpStatus[problem.status]}")
    for i in offers.keys():
        for j in demand.keys():
            print(f"Cantidad enviada desde oferta {i} a demanda {j}: {sent_quantity[i][j].varValue}")

    return render_template('table.html', total=total, rows=rows, cols=cols)

@app.route('/costominimo', methods=['POST'])
def costominimo():
    # Columnas y filas
    N = int(request.form['N'])
    M = int(request.form['M'])

    # Capturar los costos, ofertas y demandas
    costs = []
    for i in range(N):
        row = []
        for j in range(M):
            cell_name = f'cell_{i}_{j}'
            row.append(int(request.form[cell_name]))
        costs.append(row)

    #Obtener valores de ofertas y demandas
    offers = [int(request.form[f'offer_{i}_']) for i in range(N)]
    demands = [int(request.form[f'demand_{j}']) for j in range(M)]

    # Llamar a la función de costo mínimo
    result, totals, minimum_cost = calcular_costo_minimo(costs, offers, demands)

    # Guardar los datos en variables globales temporales
    temp_data['costos'] = costs
    temp_data['ofertas'] = offers
    temp_data['demandas'] = demands
    temp_data['resultado'] = result

    # Renderizar la página con los resultados
    return render_template('costominimo.html', rows=N, cols=M, costs=costs, offers=offers, demands=demands, result=result, totals=totals, minimum_cost=minimum_cost)

def calcular_costo_minimo(costs, offers, demands):
    N = len(offers)
    M = len(demands)
    
    # Crear una matriz de resultados
    result = [[0]*M for _ in range(N)]
    
    # Copiar las ofertas y demandas para no modificar las originales
    actual_offer = offers[:]
    actual_demand = demands[:]
    
    # Mientras haya ofertas y demandas
    while any(actual_offer) and any(actual_demand):
        # Encontrar la celda con el menor costo
        minimum = float('inf')
        # Inicializar los índices de la celda con el menor costo
        min_i = -1
        min_j = -1
        # Iterar sobre todas las celdas
        for i in range(N):
            for j in range(M):
                if actual_offer[i] > 0 and actual_demand[j] > 0 and costs[i][j] < minimum:
                    # Actualizar el costo mínimo y los índices de la celda
                    minimum = costs[i][j]
                    min_i = i
                    min_j = j

        # Asignar la cantidad mínima de la celda con el menor costo
        asignacion = min(actual_offer[min_i], actual_demand[min_j])
        # Asignar la cantidad mínima a la celda correspondiente
        result[min_i][min_j] = asignacion
        # Actualizar las ofertas y demandas
        actual_offer[min_i] -= asignacion
        actual_demand[min_j] -= asignacion

        # Calcular los totales por columna
        totals = [0] * M
        for j in range(M):
            for i in range(N):
                totals[j] += result[i][j] * costs[i][j]

        minimum_cost = sum([totals[j] for j in range(M)])
    return result, totals, minimum_cost

def representar_modelo_de_red(costos, ofertas, demandas, resultado):
    G = nx.DiGraph()

    # Añadir nodos de oferta
    for i in range(len(ofertas)):
        G.add_node(f'O{i+1}', demand=-ofertas[i])
    
    # Añadir nodos de demanda
    for j in range(len(demandas)):
        G.add_node(f'D{j+1}', demand=demandas[j])

    # Añadir aristas con capacidad (resultado) y costo
    for i in range(len(ofertas)):
        for j in range(len(demandas)):
            G.add_edge(f'O{i+1}', f'D{j+1}', capacity=resultado[i][j], weight=costos[i][j])

    pos = {f'O{i+1}': (0, -i) for i in range(len(ofertas))}
    pos.update({f'D{j+1}': (1, -j) for j in range(len(demandas))})

    edge_labels = {(f'O{i+1}', f'D{j+1}'): f"{resultado[i][j]} (costo: {costos[i][j]})" 
                   for i in range(len(ofertas)) for j in range(len(demandas))}

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

@app.route('/grafo')
def mostrar_grafo():
    # Recuperar los datos de las variables globales temporales
    costos = temp_data['costos']
    ofertas = temp_data['ofertas']
    demandas = temp_data['demandas']
    resultado = temp_data['resultado']

    if not costos or not ofertas or not demandas or not resultado:
        return "No hay datos para mostrar el grafo. Por favor, resuelve primero el problema de costo mínimo.", 400

    # Generar la representación del grafo
    fig = plt.figure()
    representar_modelo_de_red(costos, ofertas, demandas, resultado)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    png_output = base64.b64encode(output.getvalue()).decode('ascii')
    plt.close(fig)

    return render_template('grafo.html', grafo_img=png_output)

if __name__ == '__main__':
    app.run(debug=True)
