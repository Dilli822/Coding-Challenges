

// D3 has the attr() method to add any HTML attribute to an element, including a class name.

// The attr() method works the same way that style() does. It takes comma-separated values, and can use a callback function. 
// selection.attr("class", "container");
// Note that the class parameter will remain the same whenever you need to add a class and only the container parameter will change.


const html = `
<style>
  .bar {
    width: 25px;
    height: 100px;
    display: inline-block;
    background-color: blue;
  }
</style>
<body>
  <script>
    const dataset = [12, 31, 22, 17, 25, 18, 29, 14, 9];

    d3.select("body").selectAll("div")
      .data(dataset)
      .enter()
      .append("div")
      // Add your code below this line
      .attr("class", "bar")


      // Add your code above this line
  </script>
</body>
`;

// update the height dynamically
const html2 = `
<style>
  .bar {
    width: 25px;
    height: 920px;
    margin: 0 5px;
    display: inline-block;
    background-color: blue;
  }
</style>
<body>
  <script>
    const dataset = [12, 31, 22, 17, 25, 18, 29, 14, 9];

    d3.select("body").selectAll("div")
      .data(dataset)
      .enter()
      .append("div")
      .attr("class", "bar")
      // Add your code below this line

      .style("height", (d) => 
      d
      )
      // Add your code above this line
  </script>
</body>
`;


// Change the Presentation of a Bar Chart
// The last challenge created a bar chart, but there are a couple of formatting changes that could improve it:

// Add space between each bar to visually separate them, which is done by adding a margin to the CSS for the bar class

// Increase the height of the bars to better show the difference in values, which is done by multiplying the value by a number to scale the height

// First, add a margin of 2px to the bar class in the style tag. Next, change the callback function in the style() method so it returns a value 10 times the original data value (plus the px).

// Note: Multiplying each data point by the same constant only alters the scale. It's like zooming in, and it doesn't change the meaning of the underlying data.


const html3 = `
<style>
  .bar {
    width: 25px;
    height: 100px;
    /* Add your code below this line */
    margin: 2px;
    
    /* Add your code above this line */
    display: inline-block;
    background-color: blue;
  }
</style>
<body>
  <script>
    const dataset = [120, 310, 220, 170, 250, 180, 290, 140, 90];

    d3.select("body").selectAll("div")
      .data(dataset)
      .enter()
      .append("div")
      .attr("class", "bar")
      .style("height", (d) => (d + "px")) // Change this line
  </script>
</body>
`