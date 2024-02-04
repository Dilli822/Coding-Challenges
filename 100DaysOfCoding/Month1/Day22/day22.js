

const DynamicallySettheCoordinatesforEachBar = `

<body>
  <script>
    const dataset = [12, 31, 22, 17, 25, 18, 29, 14, 9];

    const w = 500;
    const h = 100;

    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    svg.selectAll("rect")
       .data(dataset)
       .enter()
       .append("rect")
       .attr("x", (d, i) => i * 30) // Set x attribute
       .attr("y", 0)
       .attr("width", 25)
       .attr("height", 100);
  </script>
</body>

`;


const DynamicallyChangetheHeightofEachBar = `
<body>
  <script>
    const dataset = [12, 31, 22, 17, 25, 18, 29, 14, 9];

    const w = 500;
    const h = 100;

    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    svg.selectAll("rect")
       .data(dataset)
       .enter()
       .append("rect")
       .attr("x", (d, i) => i * 30) // Set x attribute
       .attr("y", 0)
       .attr("width", 25)
       .attr("height", 100)
       .style("fill", (d, i) => i % 2 === 0 ? "red" : "blue"); // Set fill attribute based on index
  </script>
</body>

`;


// - SVG (100)
//   - Bar 1 (height: 2, y = 100 - 3 * 2 = 94)
//   - Bar 2 (height: 4, y = 100 - 3 * 4 = 88)
//   - Bar 3 (height: 1, y = 100 - 3 * 1 = 97)

// In SVG, the origin point for the coordinates is in the upper-left corner. An x coordinate of 0 places a shape on the left edge of the SVG area. A y coordinate of 0 places a shape on the top edge of the SVG area. Higher x values push the rectangle to the right. Higher y values push the rectangle down.

