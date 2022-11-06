import tkinter as tk

selection = None

def App(imgs):
    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()
    
    def callback(num):
        global selection
        selection = num
        root.quit()
        
    for i, img in enumerate(imgs):
        image = tk.PhotoImage(file=img)
        button = tk.Button(frame, image = image, command=lambda x=i: callback(x))
        
        button.pack(side=tk.LEFT)
    
    root.mainloop()
    return selection