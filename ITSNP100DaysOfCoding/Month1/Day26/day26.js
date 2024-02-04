

// The D3 methods domain() and range() set that information for your scale based on the data.

const Use_the_d3_max_and_d3_min_Functions_to_Find_Minimum_and_Maximum_Values_in_a_Dataset = `
<body>
  <script>
    const positionData = [[1, 7, -4],[6, 3, 8],[2, 9, 3]]
    // Add your code below this line

    const output = d3.max(positionData, (d) => d[2]); // Change this line
    // Add your code above this line
    d3.select("body")
      .append("h2")
      .text(output)
      
  </script>
</body>

`


const exampleData = [34, 234, 73, 90, 6, 52];
d3.min(exampleData)
d3.max(exampleData)


const locationData = [[1, 7],[6, 3],[8, 3]];
const minX = d3.min(locationData, (d) => d[0]);

// The D3 min() and max() methods are useful to help set the scale.
// The domain() method passes information to the scale about the raw data values for the plot. The range() method gives it information about the actual space on the web page for the visualization.

const range_domain = `

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D3 Simple Scale Example</title>
  <script src="https://d3js.org/d3.v5.min.js"></script>
</head>
<body>

<script>
  // Simple dataset with two values
  const dataset = [10, 50];

  // Set up an SVG container
  const svg = d3.select("body")
    .append("svg")
    .attr("width", 300)
    .attr("height", 100);

  // Set up a linear scale for the x-axis
  const xScale = d3.scaleLinear()
    .domain([0, d3.max(dataset)]) // Domain from 0 to the maximum value in the dataset
    // .range([0, 0]); // Range from 0 to the SVG width
    .range([0, 450]); // Range from 0 to the SVG width
  // Create a rectangle based on the dataset
  svg.selectAll("rect")
    .data(dataset)
    .enter()
    .append("rect")
    .attr("x", (d, i) => xScale(d)) // Use the scale for positioning
    .attr("y", 30)
    .attr("width", 20)
    .attr("height", 40)
    .attr("fill", "steelblue");
</script>

</body>
</html>

`



const dynamic_Scale = `


<body>
  <script>
    const dataset = [
                  [ 34,    78 ],
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
    // Padding between the SVG boundary and the plot
    const padding = 30;
    // Create an x and y scale
    const xScale = d3.scaleLinear()
    // passes information to the scale about the raw data values for the plot.
                    .domain([0, d3.max(dataset, (d) => d[0])])
                    //  range() method gives it information about the actual space on the web page for the visualization.
                    .range([padding, w - padding]);
console.log(xScale);
    // Add your code below this line
   
const yScale = d3.scaleLinear()
  .domain([0, d3.max(dataset, (d) => d[1])])
  //  range() of yScale should be equivalent to [470, 30]
  .range([h - padding, padding]);

    // Add your code above this line
    // here 411 is the raw data to be mapped or scale
    const output = yScale(411); // Returns 30
    d3.select("body")
      .append("h2")
      .text(output)
      
  </script>
</body>`