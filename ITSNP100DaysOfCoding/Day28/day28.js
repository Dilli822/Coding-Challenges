

// Most web APIs transfer data in a format called JSON. JSON stands for JavaScript Object Notation.

// JSON syntax looks very similar to JavaScript object literal notation. JSON has object properties and 
// their current values, sandwiched between a { and a }.

// These properties and their values are often referred to as "key-value pairs".

// However, JSON transmitted by APIs are sent as bytes, and your application receives it as a string. 
// These can be converted into JavaScript objects, but they are not JavaScript objects by default.
// The JSON.parse method parses the string and constructs the JavaScript object described by it.


// const req = new XMLHttpRequest();
// req.open("GET",'/json/cats.json',true);
// req.send();
// req.onload = function(){
//   const json = JSON.parse(req.responseText);
//   document.getElementsByClassName('message')[0].innerHTML = JSON.stringify(json);
// };