// Assuming you have a data array
const data = [10, 20, 30, 40, 50];

// Select an existing element (e.g., a paragraph)
const paragraphs = d3.select("body").selectAll("p");

// Use .data() to bind data to the selection
const update = paragraphs.data(data);

// Use .enter() to handle new data points
const enter = update.enter();

// Append a paragraph for each new data point
enter.append("p")
    .text((d) => d); // Set the text content using the data

// Update the text content for all paragraphs
update.text((d) => d);

// Remove any paragraphs that no longer have corresponding data
update.exit().remove();



const h =`
<body>
  <script>
    const dataset = [12, 31, 22, 17, 25, 18, 29, 14, 9];

    d3.select("body").selectAll("h2")
      .data(dataset)
      .enter()
      .append("h2")
    //    Add your code below this line
 .text((d) => `${d} USD`);

      // Add your code above this line
  </script>
</body>
`;


// append() method, create a new DOM element for each entry in the data set.