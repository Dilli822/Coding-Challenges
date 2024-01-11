function steamrollArray(arr) {
    // Initialize an empty array to store the flattened elements.
    let flattenedArray = [];
    // Log the initial state of the flattenedArray.
    console.log("Initial flattened array:", flattenedArray);
  
    // Define a helper function called flatten.
    function flatten(element) {
      // Check if the current element is an array.
      if (Array.isArray(element)) {
        // Log that the element is an array.
        console.log("Array.isArray method --> ", Array.isArray(element));
  
        // Loop through the elements of the array.
        for (let i = 0; i < element.length; i++) {
          // Log the length of the current element.
          console.log("Element length is -->", element.length);
  
          // Recursively call flatten on the current element.
          flatten(element[i]);
        }
      } else {
        // If the current element is not an array, push it to the flattenedArray.
        flattenedArray.push(element);
        // Log the updated state of the flattenedArray.
        console.log("Updated flattened array:", flattenedArray);
      }
    }
  
    // Call the flatten function with the input array.
    flatten(arr);
  
    // Return the final flattenedArray.
    return flattenedArray;
  }
  
  // Test case
  console.log(steamrollArray([[["a"]], [["b"]]])); // should return ["a", "b"]
  


//   steamrollArray([[["a"]], [["b"]]]) should return ["a", "b"].
// Passed: steamrollArray([1, [2], [3, [[4]]]]) should return [1, 2, 3, 4].
// Passed: steamrollArray([1, [], [3, [[4]]]]) should return [1, 3, 4].
// Passed: steamrollArray([1, {}, [3, [[4]]]]) should return [1, {}, 3, 4].
// Passed: Your solution should not use the Array.prototype.flat() or Array.prototype.flatMap() methods.
// Passed: Global variables should not be used.


// https://chat.openai.com/share/b2e17c02-0237-4c86-945c-a4c8dd2920a2