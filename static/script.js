document.addEventListener(
"DOMContentLoaded",
function(){


const inputs =
document.querySelectorAll(
"input"
);



inputs.forEach(
function(input){


input.addEventListener(
"input",
function(){


if(this.value < 0){


this.value = 0;


}


});


});


});