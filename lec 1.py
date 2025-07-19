import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import networkx as nx
import time
def tanh(x):
    return np.tanh(x)
i1, i2 = 0.05, 0.10

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55

def animate_network():
    try:
        b1 = float(b1_entry.get())
        b2 = float(b2_entry.get())

        h1_input = (i1 * w1) + (i2 * w3) + b1
        h2_input = (i1 * w2) + (i2 * w4) + b1

        h1_output = tanh(h1_input)
        h2_output = tanh(h2_input)

        o1_input = (h1_output * w5) + (h2_output * w7) + b2
        o2_input = (h1_output * w6) + (h2_output * w8) + b2

        o1_output = tanh(o1_input)
        o2_output = tanh(o2_input)
        result_label.config(text=f"Hidden Layer Outputs:\n h1 = {h1_output:.4f}, h2 = {h2_output:.4f}\n"
                                 f"Output Layer Outputs:\n o1 = {o1_output:.4f}, o2 = {o2_output:.4f}")
        G = nx.DiGraph()
        nodes = ["i1", "i2", "h1", "h2", "o1", "o2"]
        positions = {"i1": (0, 2), "i2": (0, 0), "h1": (2, 2), "h2": (2, 0), "o1": (4, 2), "o2": (4, 0)}
        edges = [
            ("i1", "h1", w1), ("i1", "h2", w2),
            ("i2", "h1", w3), ("i2", "h2", w4),
            ("h1", "o1", w5), ("h1", "o2", w6),
            ("h2", "o1", w7), ("h2", "o2", w8)
        ]
        G.add_nodes_from(nodes)
        for src, dst, weight in edges:
            G.add_edge(src, dst, weight=weight)
        plt.figure(figsize=(8, 5))
        nx.draw(G, pos=positions, with_labels=True, node_color="lightgray", edge_color="gray", node_size=2000, font_size=12)

        edge_labels = {(src, dst): f"{weight:.2f}" for src, dst, weight in edges}
        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.title("Neural Network Visualization")
        colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33"]
        for i in range(4):
            plt.clf()
            node_colors = ["yellow" if n in ["h1", "h2"] else "lightblue" for n in G.nodes()]
            edge_colors = [colors[i % len(colors)] for _ in G.edges()]

            nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=2000, font_size=12)
            nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=10, font_color="black")

            plt.pause(0.5)  # Pause to create animation effect

        plt.show()

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for b1 and b2")
root = tk.Tk()
root.title("Neural Network Visualization")
root.geometry("400x300")
root.configure(bg="#282C34")
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=5, background="#61AFEF", foreground="white")
style.configure("TLabel", font=("Arial", 12), background="#282C34", foreground="white")
style.configure("TEntry", font=("Arial", 12), padding=5)

# Labels and Entry fields for b1 and b2
ttk.Label(root, text="Enter Bias b1:").pack(pady=5)
b1_entry = ttk.Entry(root)
b1_entry.pack(pady=5)

ttk.Label(root, text="Enter Bias b2:").pack(pady=5)
b2_entry = ttk.Entry(root)
b2_entry.pack(pady=5)
calc_button = ttk.Button(root, text="Animate Neural Network", command=animate_network)
calc_button.pack(pady=10)

result_label = ttk.Label(root, text="", font=("Arial", 12))
result_label.pack()
root.mainloop()
