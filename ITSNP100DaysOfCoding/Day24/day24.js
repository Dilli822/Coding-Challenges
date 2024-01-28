const CreateaScatterplotwithSVGCircles = `
<body>
  <script>
    const dataset = [
      [34, 78],
      [109, 280],
      [310, 120],
      [79, 411],
      [420, 220],
      [233, 145],
      [333, 96],
      [222, 333],
      [78, 320],
      [21, 123]
    ];

    const w = 500;
    const h = 500;

    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    // Add your code below this line
    svg.selectAll("circle")
      .data(dataset)
      .enter()
      .append("circle");
    // Add your code above this line
  </script>
</body>

`;

// cx and cy attributes are the coordinates. 
//  cy attribute for a circle is measured from the top of the SVG,

const AddAttributestotheCircleElements = `AddAttributestotheCircleElements
<body>
  <script>
    const dataset = [
      [34, 78],
      [109, 280],
      [310, 120],
      [79, 411],
      [420, 220],
      [233, 145],
      [333, 96],
      [222, 333],
      [78, 320],
      [21, 123]
    ];
    const w = 500;
    const h = 500;
    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    // Add your code below this line
    svg.selectAll("circle")
      .data(dataset)
      .enter()
      .append("circle")
      // x co-ordinates pushes to the right side
      // whereas cy is measured from the top 
      .attr("cx", (d) => d[0])
      .attr("cy", (d) => h - d[1])  // Subtract y-coordinate from the height of the SVG
      .attr("r", 5)
      .attr("fill", "red")
    // Add your code above this line
  </script>
</body>

`;

