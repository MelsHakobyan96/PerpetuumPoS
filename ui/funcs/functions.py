def submit(form):
    if form.get("btn1"):
       return [1, 0]
    elif form.get("btn2"):
       return [0, 1]
    elif form.get("btn3"):
        return [0.5, 0.5]
    else:
        return None
