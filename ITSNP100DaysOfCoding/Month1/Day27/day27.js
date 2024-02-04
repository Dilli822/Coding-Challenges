

const Use_a_Pre_Defined_Scale_to_Place_Elements = `

<body>
  <script>
    // Dataset containing pairs of x and y values
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

    // Width and height of the SVG container
    const w = 500;
    const h = 500;

    // Padding around the SVG container
    const padding = 60;

    // Create xScale to map x values to the SVG width within the padding
    const xScale = d3.scaleLinear()
                     .domain([0, d3.max(dataset, (d) => d[0])])
                     .range([padding, w - padding]);

    // Create yScale to map y values to the SVG height within the padding
    const yScale = d3.scaleLinear()
                     .domain([0, d3.max(dataset, (d) => d[1])])
                     .range([h - padding, padding]);

    // Create an SVG container and set its width and height
    const svg = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

    // Append circles to the SVG, positioning them based on scaled x and y values
    svg.selectAll("circle")
       .data(dataset)
       .enter()
       .append("circle")
       .attr("cx", (d) => xScale(d[0]))
       .attr("cy", (d) => yScale(d[1]))
       .attr("r", 5); // Set the radius of circles to 5 units

    // Append text labels to the SVG, displaying x and y values and offsetting x values to the right
    svg.selectAll("text")
       .data(dataset)
       .enter()
       .append("text")
       .text((d) =>  (d[0] + "," + d[1]))
       .attr("x", (d) => xScale(d[0] + 10)) // Offset x values to the right by 10 units
       .attr("y", (d) => yScale(d[1])); // Set y values based on yScale
  </script>
</body>
`;




// The linear scale created using D3.js with the specified domain and range can be expressed mathematically. The general formula for a linear scale is:
// y=mx+b

// In this formula:
// y is the output value (scaled value).
// x is the input value (data point).
// m is the slope of the line, representing the scale factor.
// b is the y-intercept, representing the offset.
// In the case of the provided D3.js code:

// const linearScale = d3.scaleLinear()
//                      .domain([0, 100]) // Input data range
//                      .range([0, 500]); // Output range
// console.log(linearScale(50)); // Output: 250
// The linear scale can be expressed mathematically as:
// y = 500/ 100 x
// Here:
// x is the input data value (50).
// y is the scaled output value (250).
// So, when 
// �
// =
// 50
// x=50, the scaled value 
// �
// y is calculated as:

// �
// =
// 500
// 100
// ×
// 50
// =
// 250
// y= 
// 100
// 500
// ​
//  ×50=250

// This is consistent with the output you see in the console (250). The linear scale linearly maps the input range 
// [
// 0
// ,
// 100
// ]
// [0,100] to the output range 
// [
// 0
// ,
// 500
// ]
// [0,500].




/***
 * 
 * 
 * before
 * <svg width="300" height="200">
  <!-- Rectangle without transformation -->
  <rect x="50" y="30" width="100" height="50" fill="blue"></rect>
</svg>
after
 * 
 * <svg width="300" height="200">
  <!-- Rectangle with translation transformation -->
  <rect width="100" height="50" fill="blue" transform="translate(50, 30)"></rect>
</svg>
resulted graph
 * 
 *     Y
    |
    |                   Without Transformation
200 +------------------------------------------------
    |                        +------------------+
    |                        |                  |
150 +------------------------|                  |
    |                        |                  |
    |                        +------------------+
    |                       
100 +------------------------------------------------
    |                        With Transformation
    |                        +------------------+
    |                        |                  |
    |                        |                  |
50  +------------------------|                  |
    |                        +------------------+
    +------------------------------------------------
       0         50         100        150        200       X

 * 
 */