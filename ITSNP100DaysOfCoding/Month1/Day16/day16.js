

/** 
 * D3.js (also known as D3, 
 * short for Data-Driven Documents) is 
 * a JavaScript library for producing dynamic, 
 * interactive data visualizations in web browsers.
 * D3 allows us to chain several methods together with periods to perform a number of actions in a row
 */


// <body>
//   <script>
//     // Add your code below this line
// console.log(d3)
// d3.select("body").append("h1").text("Learning D3")

//     // Add your code above this line
//   </script>
// </body>

const HTML = `
<body>
  <ul>
    <li>Example</li>
    <li>Example</li>
    <li>Example</li>
  </ul>
  <script>
    // Add your code below this line
    //  console.log(d3.selectAll("li").text("List item"));
  d3.selectAll("li")
      .text("list item");
    // Add your code above this line
  </script>
</body>
`;


const d3 = `
<body>
  <h2>New Title</h2>
  <h2>New Title</h2>
  <h2>New Title</h2>
  <script>
    // Add your code below this line
    const dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    d3.select("body").selectAll("h2")
      .data(dataset)
      .enter()
      .append("h2")
      .text("New Title");
    // Add your code above this line
  </script>
</body>
`
