function convertHTML(str) {
    // Define a mapping of characters to their HTML entities
    const htmlEntities = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&apos;'
    };
  
    // Check if the string includes any of the characters to be replaced
    if (/[&<>"']/.test(str)) {
      // Replace the characters in the string using the mapping
      str = str.replace(/[&<>"']/g, match => htmlEntities[match]);
      console.log(str);  // You can log the modified string here if needed
    }
  
    // Return the modified or original string
    return str;
  }
  
  // Test cases
  console.log(convertHTML("Dolce & Gabbana")); // Dolce &amp; Gabbana
  console.log(convertHTML("Hamburgers < Pizza < Tacos")); // Hamburgers &lt; Pizza &lt; Tacos
  console.log(convertHTML("Sixty > twelve")); // Sixty &gt; twelve
  console.log(convertHTML('Stuff in "quotation marks"')); // Stuff in &quot;quotation marks&quot;
  console.log(convertHTML("Schindler's List")); // Schindler&apos;s List
  console.log(convertHTML("<>")); // &lt;&gt;
  console.log(convertHTML("abc")); // abc
  
// const originalString = 'Hello, world!';
// const replacedString = originalString.replace('world', 'there');
// console.log(replacedString); // Output: Hello, there!

const products = [
    {
      "id": 1,
      "product_name": "Panasonic 64inch",
      "total_units": 30,
      "total_price": 3434340
    },
    {
      "id": 1,
      "product_name": "JBL Charge 5",
      "total_units": 30,
      "total_price": 3434340
    },
    {
      "id": 1,
      "product_name": "Beats Solo3 Wireless",
      "total_units": 12,
      "total_price": 60000
    }
  ];
  
  // Function to update total units and total prices of products
  const BeforeProducts = (products, totalUnitsArray, totalPriceArray) => {
    products.forEach((product, index) => {
      product.total_units = totalUnitsArray[index];
      product.total_price = totalPriceArray[index];
    });
  };
  console.log("brefore", products); // Output the updated products array
  
  
  const totalUnitsArray = [1, 1, 1]; // Array indicating total units for each product
  const totalPriceArray = [2.00, 5.00, 1.00]; // Array indicating total price for each product
  
  // Function to update total units and total prices of products
  const updateProducts = (products, totalUnitsArray, totalPriceArray) => {
    products.forEach((product, index) => {
      product.total_units = totalUnitsArray[index];
      product.total_price = totalPriceArray[index];
    });
  };
  
  // Update the products array
  updateProducts(products, totalUnitsArray, totalPriceArray);
  
  console.log("aftre user request ", products); // Output the updated products array


// #OUTPUTS
// "Hamburgers &lt; Pizza &lt; Tacos"
// "Sixty &gt; twelve"
// "Stuff in &quot;quotation marks&quot;"
// "Schindler&apos;s List"
// "&lt;&gt;"
// "abc"
// "Dolce &amp; Gabbana"
// "Hamburgers &lt; Pizza &lt; Tacos"
// "Sixty &gt; twelve"
// "Stuff in &quot;quotation marks&quot;"
// "Schindler&apos;s List"
// "&lt;&gt;"
// "abc"
// // before updating
// // [object Array] (3)
// [// [object Object] 
// {
//   "id": 1,
//   "product_name": "Panasonic 64inch",
//   "total_units": 30,
//   "total_price": 3434340
// },// [object Object] 
// {
//   "id": 1,
//   "product_name": "Beats Solo3 Wireless",
//   "total_units": 12,
//   "total_price": 60000
// }]

// "aftre user request " // [object Array] (3)
// [// [object Object] 
// {
//   "id": 1,
//   "product_name": "Panasonic 64inch",
//   "total_units": 1,
//   "total_price": 2
// },// [object Object] 
// {
//   "id": 1,
//   "product_name": "JBL Charge 5",
//   "total_units": 1,
//   "total_price": 5
// },// [object Object] 
// {
//   "id": 1,
//   "product_name": "Beats Solo3 Wireless",
//   "total_units": 1,
//   "total_price": 1
// }]

// // [object Array] (3)
