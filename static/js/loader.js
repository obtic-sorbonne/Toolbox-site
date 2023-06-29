function showloader(cl) {
	var loader = document.getElementsByClassName(cl)
	loader[0].style.display = "block";
	loader[0].scrollIntoView();
	hideLoadingDiv(loader[0]);
}

function hideLoadingDiv(elem) {
  setTimeout(function(){
    elem.style.display = 'none';
  }, 4000)
}
