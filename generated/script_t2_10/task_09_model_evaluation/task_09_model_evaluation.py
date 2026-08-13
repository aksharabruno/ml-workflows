from dependency import *  # noqa: F401,F403


def model_evaluation_9(lr, transformer):
    # ── Show all matplotlib figures that were built earlier ──────────────────────
    plt.show()

    # ── Tkinter UI ───────────────────────────────────────────────────────────────
    def launch_ui():
        root = tk.Tk()
        root.title("🚗  Car Selling Price Predictor")
        root.resizable(False, False)

        # ── Colour palette ────────────────────────────────────────────────────────
        BG       = "#1e1e2e"
        CARD     = "#2a2a3e"
        ACCENT   = "#7c6af7"
        FG       = "#cdd6f4"
        FG_DIM   = "#a6adc8"
        ENTRY_BG = "#313244"
        BTN_FG   = "#ffffff"

        root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel",      background=CARD, foreground=FG,     font=("Segoe UI", 10))
        style.configure("Title.TLabel",background=BG,   foreground=ACCENT, font=("Segoe UI", 15, "bold"))
        style.configure("Sub.TLabel",  background=BG,   foreground=FG_DIM, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=CARD)
        style.configure("TCombobox",   fieldbackground=ENTRY_BG, foreground=FG, background=CARD,
                        selectbackground=ENTRY_BG, selectforeground=FG, font=("Segoe UI", 10))
        style.map("TCombobox", fieldbackground=[("readonly", ENTRY_BG)])

        # ── Header ────────────────────────────────────────────────────────────────
        header = tk.Frame(root, bg=BG, pady=16)
        header.pack(fill="x", padx=24)
        ttk.Label(header, text="Car Selling Price Predictor", style="Title.TLabel").pack()
        ttk.Label(header, text="Linear Regression  •  Enter details below",
                  style="Sub.TLabel").pack(pady=(2, 0))

        # ── Card frame ────────────────────────────────────────────────────────────
        card = ttk.Frame(root, style="Card.TFrame", padding=20)
        card.pack(padx=24, pady=(0, 16), fill="both")

        def make_label(row, text):
            lbl = ttk.Label(card, text=text)
            lbl.grid(row=row, column=0, sticky="w", padx=(4, 16), pady=6)

        def entry_widget(row, default):
            var = tk.StringVar(value=str(default))
            e = tk.Entry(card, textvariable=var, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                         relief="flat", font=("Segoe UI", 10), width=22,
                         highlightthickness=1, highlightbackground=ACCENT,
                         highlightcolor=ACCENT)
            e.grid(row=row, column=1, sticky="ew", padx=4, pady=6)
            return var

        def combo_widget(row, options, default):
            var = tk.StringVar(value=default)
            cb = ttk.Combobox(card, textvariable=var, values=options,
                              state="readonly", width=20, font=("Segoe UI", 10))
            cb.grid(row=row, column=1, sticky="ew", padx=4, pady=6)
            return var

        card.columnconfigure(1, weight=1)

        fields = [
            ("Year (2003 – 2023):",    lambda r: entry_widget(r, 2015)),
            ("Present Price (Lakhs):", lambda r: entry_widget(r, 5.0)),
            ("Kms Driven:",            lambda r: entry_widget(r, 30000)),
            ("Fuel Type:",             lambda r: combo_widget(r, ["Petrol","Diesel","CNG"], "Petrol")),
            ("Seller Type:",           lambda r: combo_widget(r, ["Dealer","Individual"], "Dealer")),
            ("Transmission:",          lambda r: combo_widget(r, ["Manual","Automatic"], "Manual")),
            ("Owner (0 – 3):",         lambda r: entry_widget(r, 0)),
        ]

        vars_ = []
        for i, (label_text, widget_fn) in enumerate(fields):
            make_label(i, label_text)
            vars_.append(widget_fn(i))

        # ── Result label ──────────────────────────────────────────────────────────
        result_var = tk.StringVar(value="")
        result_lbl = tk.Label(root, textvariable=result_var,
                              bg=BG, fg=ACCENT,
                              font=("Segoe UI", 13, "bold"), pady=4)
        result_lbl.pack()

        # ── Predict button ────────────────────────────────────────────────────────
        def on_predict():
            try:
                row = pd.DataFrame([[
                    int(vars_[0].get()),
                    float(vars_[1].get()),
                    int(vars_[2].get()),
                    vars_[3].get(),
                    vars_[4].get(),
                    vars_[5].get(),
                    int(vars_[6].get()),
                ]], columns=['Year','Present_Price','Kms_Driven',
                             'Fuel_Type','Seller_Type','Transmission','Owner'])

                transformed = transformer.transform(row)
                price = lr.predict(transformed)[0]
                result_var.set(f"✅  Predicted Selling Price:  ₹ {price:.2f} Lakhs")
            except ValueError as ve:
                messagebox.showerror("Input Error", f"Please check your inputs.\n\n{ve}")
            except Exception as ex:
                messagebox.showerror("Error", str(ex))

        btn = tk.Button(root, text="  Predict Selling Price  ",
                        command=on_predict,
                        bg=ACCENT, fg=BTN_FG,
                        font=("Segoe UI", 11, "bold"),
                        relief="flat", cursor="hand2",
                        padx=12, pady=8,
                        activebackground="#6a5ce0", activeforeground=BTN_FG)
        btn.pack(pady=(0, 20))

        root.mainloop()

    launch_ui()
