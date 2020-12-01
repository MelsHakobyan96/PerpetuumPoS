function btn_click() {
  if (this.id == "btn1") {
     console.log([1, 0])
     return [1, 0]
  } else if (this.id == "btn2"){
    console.log([0, 1])
     return [0, 1]
  } else if (this.id == "btn3"){
    console.log([0.5, 0.5])
     return [0.5, 0.5]
  } else if (this.id == "btn4"){
    console.log(null)
     return null
  }
}

document.getElementById('btn1').onclick = btn_click;
document.getElementById('btn2').onclick = btn_click;
document.getElementById('btn3').onclick = btn_click;
document.getElementById('btn4').onclick = btn_click;
