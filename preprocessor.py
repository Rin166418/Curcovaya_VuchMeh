import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

import tkinter
print(tkinter.TkVersion)

PASTEL_BG = "#e6f0ff"
PASTEL_BUTTON = "#99c2ff"
PASTEL_HOVER = "#b3d1ff"
PASTEL_TREE_BG = "#cce0ff"
PASTEL_TREE_ALT = "#e6f2ff"
PASTEL_TREE_BORDER = "#4da6ff"

DEFAULT_A = 1.0
DEFAULT_E = 1.0
DEFAULT_SIGMA = 1.0

#Оформление
def create_rounded_rect(canvas, x1, y1, x2, y2, radius=10, **kwargs):
    points = [
        x1+radius, y1,
        x2-radius, y1,
        x2, y1, x2, y1+radius,
        x2, y2-radius,
        x2, y2, x2-radius, y2,
        x1+radius, y2,
        x1, y2, x1, y2-radius,
        x1, y1+radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Frame):
    def __init__(self, master, text, command=None, width=150, height=28):
        super().__init__(master, bg=PASTEL_BG)
        self.command = command
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0, bg=PASTEL_BG)
        self.canvas.pack()
        self.rect = create_rounded_rect(self.canvas, 2, 2, width-2, height-2, radius=8,
                                        outline="#4da6ff", width=2, fill=PASTEL_BUTTON)
        self.text = self.canvas.create_text(width/2, height/2, text=text, font=("Arial", 10, "bold"))
        self.canvas.bind("<Button-1>", lambda e: self.on_click())
        self.canvas.bind("<Enter>", lambda e: self.canvas.itemconfig(self.rect, fill=PASTEL_HOVER))
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfig(self.rect, fill=PASTEL_BUTTON))

    def on_click(self):
        if self.command:
            self.command()

class PreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Плоская стержневая система")
        self.root.geometry("1600x800")
        self.root.configure(bg=PASTEL_BG)

        self.nodes = []             
        self.elements = []  
        self.supports = []   
        self.forces = [] 
        self.element_forces = []
        self.scale = 1.0

        self.create_ui()
        self.draw_system()

    #Окно
    def create_ui(self):
        control_container = ttk.Frame(self.root)
        control_container.pack(side=tk.LEFT, fill=tk.Y)

        canvas_ctrl = tk.Canvas(control_container, width=280, bg=PASTEL_BG, highlightthickness=0)
        canvas_ctrl.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas_ctrl.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)

        self.control_frame = tk.Frame(canvas_ctrl, bg=PASTEL_BG)
        canvas_ctrl.create_window((0,0), window=self.control_frame, anchor="nw")
        self.control_frame.bind("<Configure>", lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))

        tk.Label(self.control_frame, text="Добавление узла", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=5)
        tk.Label(self.control_frame, text="Координата X:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_node = ttk.Entry(self.control_frame,width=15)
        self.entry_node.pack(anchor=tk.W)
        RoundedButton(self.control_frame, "Добавить узел", self.add_node).pack(fill=tk.X, pady=2)
        RoundedButton(self.control_frame, "Удалить узел", self.delete_node).pack(fill=tk.X, pady=2)

        tk.Label(self.control_frame, text="Добавление стержня", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=5)
        tk.Label(self.control_frame, text="Начальный узел n1:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.combo_n1 = ttk.Combobox(self.control_frame, width=5, state='readonly')
        self.combo_n1.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Конечный узел n2:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.combo_n2 = ttk.Combobox(self.control_frame, width=5, state='readonly')
        self.combo_n2.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Площадь A:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_A = ttk.Entry(self.control_frame,width=10)
        self.entry_A.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Модуль упругости E:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_E = ttk.Entry(self.control_frame,width=10)
        self.entry_E.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Допускаемое напряжение σ:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_sigma = ttk.Entry(self.control_frame,width=10)
        self.entry_sigma.pack(anchor=tk.W)
        RoundedButton(self.control_frame, "Добавить стержень", self.add_element).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Удалить стержень", self.delete_element).pack(fill=tk.X,pady=2)

        tk.Label(self.control_frame, text="Распределённая нагрузка q", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=5)
        tk.Label(self.control_frame, text="Выберите стержень:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.combo_q_element = ttk.Combobox(self.control_frame, width=10, state='readonly')
        self.combo_q_element.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Значение q:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_q = ttk.Entry(self.control_frame,width=10)
        self.entry_q.pack(anchor=tk.W)
        RoundedButton(self.control_frame, "Добавить нагрузку q", self.add_q).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Удалить нагрузку q", self.delete_q).pack(fill=tk.X,pady=2)

        tk.Label(self.control_frame, text="Добавление опоры", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=5)
        tk.Label(self.control_frame, text="Выберите узел:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.combo_support = ttk.Combobox(self.control_frame,width=5,state='readonly')
        self.combo_support.pack(anchor=tk.W)
        RoundedButton(self.control_frame, "Добавить опору", self.add_support).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Удалить опору", self.delete_support).pack(fill=tk.X,pady=2)

        tk.Label(self.control_frame, text="Силы F на узлах", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=5)
        tk.Label(self.control_frame, text="Выберите узел:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.combo_force_node = ttk.Combobox(self.control_frame,width=5,state='readonly')
        self.combo_force_node.pack(anchor=tk.W)
        tk.Label(self.control_frame, text="Значение F:", bg=PASTEL_BG).pack(anchor=tk.W)
        self.entry_F = ttk.Entry(self.control_frame,width=10)
        self.entry_F.pack(anchor=tk.W)
        RoundedButton(self.control_frame, "Добавить силу F", self.add_force).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Удалить силу F", self.delete_force).pack(fill=tk.X,pady=2)

        tk.Label(self.control_frame, text="Управление проектом", font=("Arial",11,"bold"), bg=PASTEL_BG).pack(pady=10)
        RoundedButton(self.control_frame, "Очистить всё", self.clear_all).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Масштаб +", lambda:self.scale_canvas(1.2)).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Масштаб -", lambda:self.scale_canvas(0.8)).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Сброс масштаба", self.reset_scale).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Экспорт JSON", self.export_json).pack(fill=tk.X,pady=2)
        RoundedButton(self.control_frame, "Импорт JSON", self.import_json).pack(fill=tk.X,pady=2)

        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.scale_canvas(1.1))
        self.canvas.bind("<Button-5>", lambda e: self.scale_canvas(0.9))

        self.tab_control = ttk.Notebook(right_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview",
                        background=PASTEL_TREE_BG,
                        fieldbackground=PASTEL_TREE_BG,
                        foreground="black",
                        rowheight=25,
                        font=("Arial", 10))
        style.map("Treeview", background=[('selected', '#66b3ff')])
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background=PASTEL_BUTTON, foreground="black")

        # --- Вкладки ---
        self.tree_nodes = ttk.Treeview(self.tab_control, columns=("X"), show='headings')
        self.tree_nodes.heading("X", text="X")
        self.tree_nodes.pack(expand=True, fill=tk.BOTH)
        self.tab_control.add(self.tree_nodes, text="Узлы")

        self.tree_elements = ttk.Treeview(self.tab_control, columns=("nodes","L","A","E","sigma"), show='headings')
        for col in self.tree_elements['columns']: self.tree_elements.heading(col, text=col)
        self.tab_control.add(self.tree_elements, text="Стержни")

        self.tree_q = ttk.Treeview(self.tab_control, columns=("nodes","q"), show='headings')
        for col in self.tree_q['columns']: self.tree_q.heading(col, text=col)
        self.tab_control.add(self.tree_q, text="Нагрузки q")

        self.tree_supports = ttk.Treeview(self.tab_control, columns=("node"), show='headings')
        self.tree_supports.heading("node", text="Узел")
        self.tab_control.add(self.tree_supports, text="Опоры")

        self.tree_forces = ttk.Treeview(self.tab_control, columns=("node","F"), show='headings')
        for col in self.tree_forces['columns']: self.tree_forces.heading(col, text=col)
        self.tab_control.add(self.tree_forces, text="Силы F")


    #Обновление окошек с выбором
    def update_comboboxes(self):
        node_str = [str(i+1) for i in range(len(self.nodes))] if self.nodes else []
        self.combo_n1['values'] = node_str
        self.combo_n2['values'] = node_str
        self.combo_support['values'] = node_str
        self.combo_force_node['values'] = node_str

        if self.elements:
            sorted_with_index = sorted(enumerate(self.elements), key=lambda x: min(self.nodes[x[1]['nodes'][0]-1], self.nodes[x[1]['nodes'][1]-1]))
            self.sorted_elements_for_combo = sorted_with_index
            element_str = [f"{e['nodes'][0]}-{e['nodes'][1]}" for idx,e in sorted_with_index]
        else:
            self.sorted_elements_for_combo = []
            element_str = []

        self.combo_q_element['values'] = element_str

        self.combo_n1.set('')
        self.combo_n2.set('')
        self.combo_q_element.set('')
        self.combo_support.set('')
        self.combo_force_node.set('')


    #Обновление табличек
    def refresh_lists(self):
            self.tree_nodes.delete(*self.tree_nodes.get_children())
            for i,x in enumerate(sorted(self.nodes), start=1):
                self.tree_nodes.insert("", tk.END, values=(f"{x:.2f}",))

            self.tree_elements.delete(*self.tree_elements.get_children())
            sorted_elements = sorted(self.elements, key=lambda e: min(self.nodes[e['nodes'][0]-1], self.nodes[e['nodes'][1]-1]))
            for e in sorted_elements:
                n1,n2 = e['nodes']
                self.tree_elements.insert("", tk.END, values=(f"{n1}-{n2}", f"{e['L']:.2f}", f"{e['A']:.2f}", f"{e['E']:.2e}", f"{e['sigma']:.2e}"))

            self.tree_q.delete(*self.tree_q.get_children())
            sorted_q = []
            for idx, q_val in self.element_forces:
                e = self.elements[idx]
                sorted_q.append((min(self.nodes[e['nodes'][0]-1], self.nodes[e['nodes'][1]-1]), idx, q_val))
            sorted_q.sort()
            for _, idx, q_val in sorted_q:
                e = self.elements[idx]
                n1,n2 = e['nodes']
                self.tree_q.insert("", tk.END, values=(f"{n1}-{n2}", f"{q_val:.2f}"))

            self.tree_supports.delete(*self.tree_supports.get_children())
            for n in sorted(self.supports):
                self.tree_supports.insert("", tk.END, values=(n,))

            self.tree_forces.delete(*self.tree_forces.get_children())
            for n,F in sorted(self.forces, key=lambda x:x[0]):
                self.tree_forces.insert("", tk.END, values=(n,F))
    #Добавление
    def add_node(self):
        try: 
            x = float(self.entry_node.get())
        except: 
            messagebox.showerror("Ошибка","Введите число для X")
            return
        if x in self.nodes: 
            messagebox.showerror("Ошибка","Узел с такой координатой уже существует!") 
            return
        self.nodes.append(x)
        self.nodes.sort()
        self.update_comboboxes()
        self.entry_node.delete(0,tk.END)
        self.refresh_lists()
        self.draw_system()

    def add_element(self):
        try:
            n1 = int(self.combo_n1.get())
            n2 = int(self.combo_n2.get())
        except:
            messagebox.showerror("Ошибка","Выберите начальный и конечный узел") 
            return

        try:
            A = float(self.entry_A.get().strip()) if self.entry_A.get().strip() else DEFAULT_A
            if A <= 0: 
                messagebox.showerror("Ошибка","Площадь A должна быть больше 0")
                return
        except:
            messagebox.showerror("Ошибка","Некорректное значение A")
            return

        try:
            E = float(self.entry_E.get().strip()) if self.entry_E.get().strip() else DEFAULT_E
            if E <= 0: 
                messagebox.showerror("Ошибка","Модуль упругости E должен быть больше 0")
                return
        except:
            messagebox.showerror("Ошибка","Некорректное значение E")
            return

        try:
            sigma = float(self.entry_sigma.get().strip()) if self.entry_sigma.get().strip() else DEFAULT_SIGMA
            if sigma <= 0: 
                messagebox.showerror("Ошибка","Допускаемое напряжение σ должно быть больше 0")
                return
        except:
            messagebox.showerror("Ошибка","Некорректное значение σ")
            return

        if n1 == n2:
            messagebox.showerror("Ошибка","Начальный и конечный узлы не могут совпадать")
            return

        x1_new = min(self.nodes[n1-1], self.nodes[n2-1])
        x2_new = max(self.nodes[n1-1], self.nodes[n2-1])

        for e in self.elements:
            x1_e = min(self.nodes[e['nodes'][0]-1], self.nodes[e['nodes'][1]-1])
            x2_e = max(self.nodes[e['nodes'][0]-1], self.nodes[e['nodes'][1]-1])
            if max(x1_e, x1_new) < min(x2_e, x2_new):
                messagebox.showerror("Ошибка","Новый стержень пересекается с существующим!")
                return

        for i, x in enumerate(self.nodes, start=1):
            if i != n1 and i != n2:
                if x1_new < x < x2_new:
                    messagebox.showerror("Ошибка", f"Стержень не может пересекать узел {i}")
                    return

        L = abs(self.nodes[n2-1] - self.nodes[n1-1])
        self.elements.append({'nodes': (n1, n2), 'A': A, 'E': E, 'sigma': sigma, 'L': L})

        self.entry_A.delete(0, tk.END)
        self.entry_E.delete(0, tk.END)
        self.entry_sigma.delete(0, tk.END)

        self.combo_n1.set('')
        self.combo_n2.set('')

        self.update_comboboxes()
        self.refresh_lists()
        self.draw_system()

    def add_q(self):
        try:
            combo_idx = self.combo_q_element.current()
            if combo_idx == -1:
                raise ValueError
            idx, _ = self.sorted_elements_for_combo[combo_idx]  # реальный индекс в self.elements
            q_val = float(self.entry_q.get())
        except:
            messagebox.showerror("Ошибка","Некорректные данные")
            self.entry_q.delete(0, tk.END)
            self.combo_q_element.set('')
            return

        if q_val == 0:
            messagebox.showerror("Ошибка","Нельзя добавлять распределённую нагрузку q=0")
            self.entry_q.delete(0, tk.END)
            self.combo_q_element.set('')
            return

        if any(e_idx == idx for e_idx, _ in self.element_forces):
            messagebox.showerror("Ошибка","На этом стержне уже есть распределённая нагрузка q")
            self.entry_q.delete(0, tk.END)
            self.combo_q_element.set('')
            return

        self.element_forces.append((idx, q_val))
        self.entry_q.delete(0, tk.END)
        self.combo_q_element.set('')
        self.refresh_lists()
        self.draw_system()

    def add_support(self):
        try:
            n = int(self.combo_support.get())
        except:
            messagebox.showerror("Ошибка","Выберите узел для опоры")
            return

        if n in self.supports:
            messagebox.showerror("Ошибка", "На этом узле уже есть опора")
            return

        self.supports.append(n)
        self.combo_support.set('')
        self.refresh_lists()
        self.draw_system()

    def add_force(self):
        try:
            n = int(self.combo_force_node.get())
            F = float(self.entry_F.get())
        except:
            messagebox.showerror("Ошибка","Некорректные данные")
            self.entry_F.delete(0, tk.END)
            self.combo_force_node.set('')
            return

        if F == 0:
            messagebox.showerror("Ошибка", "Сила F не может быть равна 0")
            self.entry_F.delete(0, tk.END)
            self.combo_force_node.set('')
            return

        if any(f[0] == n for f in self.forces):
            messagebox.showerror("Ошибка","На узле уже есть сила")
            self.entry_F.delete(0, tk.END)
            self.combo_force_node.set('')
            return

        self.forces.append((n,F))
        self.entry_F.delete(0, tk.END)
        self.combo_force_node.set('')
        self.refresh_lists()
        self.draw_system()

    #Удаление
    def delete_node(self):
        selected = self.tree_nodes.selection()
        if not selected:
            messagebox.showwarning("Удаление", "Выберите узел")
            return

        tree_idx = self.tree_nodes.index(selected[0])
        sorted_nodes = sorted(self.nodes)
        node_to_remove = sorted_nodes[tree_idx]
        for e in self.elements:
            n1, n2 = e['nodes']
            if self.nodes[n1-1] == node_to_remove or self.nodes[n2-1] == node_to_remove:
                messagebox.showerror("Ошибка", "Нельзя удалить узел, входящий в стержень")
                return

        self.nodes.remove(node_to_remove)
        self.update_comboboxes()
        self.refresh_lists()
        self.draw_system()

    def delete_element(self):
        selected = self.tree_elements.selection()
        if not selected:
            messagebox.showwarning("Удаление","Выберите стержень")
            return
        idx = self.tree_elements.index(selected[0])
        sorted_elements = sorted(enumerate(self.elements), key=lambda x: min(self.nodes[x[1]['nodes'][0]-1], self.nodes[x[1]['nodes'][1]-1]))
        real_idx, _ = sorted_elements[idx]
        del self.elements[real_idx]
        self.element_forces = [(i-(1 if i>real_idx else 0), q) for i,q in self.element_forces if i!=real_idx]
        self.update_comboboxes()
        self.refresh_lists()
        self.draw_system()

    def delete_q(self):
        selected = self.tree_q.selection()
        if not selected:
            messagebox.showwarning("Удаление","Выберите нагрузку q")
            return
        idx = self.tree_q.index(selected[0])
        sorted_q = sorted(self.element_forces, key=lambda x: min(self.nodes[self.elements[x[0]]['nodes'][0]-1], self.nodes[self.elements[x[0]]['nodes'][1]-1]))
        real_idx = self.element_forces.index(sorted_q[idx])
        del self.element_forces[real_idx]
        self.refresh_lists()
        self.draw_system()

    def delete_support(self):
        selected = self.tree_supports.selection()
        if not selected:
            messagebox.showwarning("Удаление","Выберите опору")
            return

        tree_idx = self.tree_supports.index(selected[0])
        sorted_supports = sorted(self.supports)
        support_to_remove = sorted_supports[tree_idx]

        self.supports.remove(support_to_remove)
        self.refresh_lists()
        self.draw_system()

    def delete_force(self):
        selected = self.tree_forces.selection()
        if not selected:
            messagebox.showwarning("Удаление","Выберите силу")
            return

        tree_idx = self.tree_forces.index(selected[0])
        sorted_forces = sorted(self.forces, key=lambda x: x[0])
        force_to_remove = sorted_forces[tree_idx]

        self.forces = [f for f in self.forces if f != force_to_remove]

        self.refresh_lists()
        self.draw_system()


    #Очистить все
    def clear_all(self):
        if messagebox.askyesno("Подтверждение","Очистить всё?"):
            self.nodes.clear()
            self.elements.clear()
            self.supports.clear()
            self.forces.clear()
            self.element_forces.clear()

            self.combo_n1.set('')
            self.combo_n2.set('')
            self.combo_q_element.set('')
            self.combo_support.set('')
            self.combo_force_node.set('')

            self.update_comboboxes()
            self.refresh_lists()
            self.canvas.delete("all")


    #Масштаб
    def scale_canvas(self,factor):
        self.scale *= factor
        self.draw_system()

    def reset_scale(self):
        self.scale = 1.0
        self.draw_system()

    def on_mousewheel(self,event):
        self.scale_canvas(1.1 if event.delta>0 else 0.9)

    #Рисуем
    def draw_system(self):
        self.canvas.delete("all")
        if not self.nodes: return
        w,h = self.canvas.winfo_width(), self.canvas.winfo_height()
        margin = 80
        x_min, x_max = min(self.nodes), max(self.nodes)
        scale = (w-2*margin)/(x_max-x_min+1e-9)*self.scale
        y_mid = h/2

        self.canvas.create_line(margin,y_mid+50,w-margin,y_mid+50,width=2,arrow=tk.LAST)

        for idx,e in enumerate(self.elements):
            n1,n2 = e['nodes']
            x1 = margin + (self.nodes[n1-1]-x_min)*scale
            x2 = margin + (self.nodes[n2-1]-x_min)*scale
            self.canvas.create_line(x1,y_mid,x2,y_mid,width=3)
            self.canvas.create_text((x1+x2)/2, y_mid-15, text=f"{idx+1}", fill="black", font=("Arial",9))

        for idx,q_val in self.element_forces:
            e = self.elements[idx]
            x1 = margin + (self.nodes[e['nodes'][0]-1]-x_min)*scale
            x2 = margin + (self.nodes[e['nodes'][1]-1]-x_min)*scale
            num_arrows = max(2, int(abs(x2-x1)/20))
            step = (x2-x1)/num_arrows
            for i_arrow in range(num_arrows):
                px = x1 + i_arrow*step
                if q_val>0: self.canvas.create_line(px,y_mid+20,px+10,y_mid+20,fill="blue",arrow=tk.LAST,width=2)
                else: self.canvas.create_line(px,y_mid+20,px-10,y_mid+20,fill="blue",arrow=tk.LAST,width=2)
            self.canvas.create_text((x1+x2)/2, y_mid+35, text=f"q={q_val}", fill="blue", font=("Arial",9))

        for i,x in enumerate(self.nodes,start=1):
            X = margin + (x-x_min)*scale
            self.canvas.create_oval(X-3,y_mid-3,X+3,y_mid+3,fill="black")
            self.canvas.create_text(X, y_mid-10, text=f"{i}")
            self.canvas.create_text(X,y_mid+60,text=f"{x:.2f}",font=("Arial",9))

        for n in self.supports:
            X = margin + (self.nodes[n-1]-x_min)*scale
            self.canvas.create_line(X,y_mid+5,X,y_mid+35,fill="red",width=3)
            for i in range(5): self.canvas.create_line(X-10,y_mid+20+i*3,X+10,y_mid+20+i*3,fill="red")

        for n,F in self.forces:
            X = margin + (self.nodes[n-1]-x_min)*scale
            arrow_len=30
            if F>0: self.canvas.create_line(X,y_mid-30,X+arrow_len,y_mid-30,fill="green",arrow=tk.LAST,width=2)
            else: self.canvas.create_line(X,y_mid-30,X-arrow_len,y_mid-30,fill="green",arrow=tk.LAST,width=2)
            self.canvas.create_text(X,y_mid-40,text=f"F={F}",fill="green",font=("Arial",9))
    # Экспорт
    def export_json(self):
        data = {'nodes':self.nodes,'elements':self.elements,'supports':self.supports,'forces':self.forces,'element_forces':self.element_forces}
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if file_path:
            with open(file_path,"w") as f: json.dump(data,f)
            messagebox.showinfo("Экспорт","Проект сохранён.")

    #Импорт
    def import_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if file_path:
            with open(file_path,"r") as f:
                data = json.load(f)

            self.nodes = data.get('nodes', [])
            self.elements = data.get('elements', [])
            self.supports = data.get('supports', [])
            self.forces = data.get('forces', [])

            self.element_forces.clear()
            for idx, q in data.get('element_forces', []):
                if 0 <= idx < len(self.elements) and q != 0:
                    self.element_forces.append((idx, q))

            self.update_comboboxes()
            self.refresh_lists()
            self.draw_system()
            messagebox.showinfo("Импорт","Проект загружен.")


if __name__=="__main__":
    root = tk.Tk()
    root.iconbitmap("icon.ico")
    app = PreprocessorApp(root)
    root.mainloop()
