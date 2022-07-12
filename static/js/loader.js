function showloader(iddiv) {
	var elem = document.getElementById(iddiv)
	elem.style.display = "block";
	elem.scrollIntoView();
	hideLoadingDiv(iddiv);
}

function hideLoadingDiv(iddiv) {
  setTimeout(function(){
    document.getElementById(iddiv).style.display = 'none';
  }, 4000)
}
