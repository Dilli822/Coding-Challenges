

const Add_Labels_to_Scatter_Plot_Circles = `
<body>
  <script>
    const dataset = [
               %   [ 34,    78 ],
                  [ 109,   280 ],
                  [ 310,   120 ],
                  [ 79,    411 ],
                  [ 420,   220 ],
                  [ 233,   145 ],
                  [ 333,   96 ],
                  [ 222,   333 ],
                  [ 78,    320 ],
                  [ 21,    123 ]
                ];


    const w = 500;
    const h = 500;

    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    svg.selectAll("circle")
       .data(dataset)
       .enter()
       .append("circle")
       .attr("cx", (d, i) => d[0])
       .attr("cy", (d, i) => h - d[1])
       .attr("r", 15)
       .attr("fill", "red");

    svg.selectAll("text")
       .data(dataset)
       .enter()
       .append("text")
       // Add your code below this line
       .attr("x", (d, i)=> d[0] + 15)
       .attr("y", (d, i)=> h - d[1])
       // Display comma-separated values as text
        .text((d) => `${d[0]}, ${d[1]}`) 
        // Set font size for the text
      .attr("font-size", 15) 
      .attr("fill", "black"); 

       // Add your code above this line
  </script>
</body>
`;


// The bar and scatter plot charts both plotted data directly onto the SVG. However, if the height of a bar or one of the data points were larger than the SVG height or width values, it would go outside the SVG area.
// Create a Linear Scale with D3
// D3 has several scale types. For a linear scale (usually used with quantitative data), there is the D3 method scaleLinear():

// It's unlikely you would plot raw data as-is. Before plotting it, you set the scale for your entire data set, so that the x and y values fit your SVG width and height.
const Create_a_Linear_Scale_with_D3 = `
<body>
  <script>
    // Add your code below this line
    const scale = d3.scaleLinear()
    .domain([250, 500])
    .range([10, 150])

    // Add your code above this line
    const output = scale(50);
    d3.select("body")
      .append("h2")
      .text(output);
  </script>
</body>
`;



// 
// 

const Set_a_Domain_and_a_Range_on_a_Scale = `

<body>
  <script>
    // Add your code below this line
    const scale = d3.scaleLinear()
    .domain([250, 500])
    .range([10, 150])

    // Add your code above this line
    const output = scale(50);
    d3.select("body")
      .append("h2")
      .text(output);
  </script>
</body>
`