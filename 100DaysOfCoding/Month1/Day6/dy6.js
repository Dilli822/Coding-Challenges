function dropElements(arr, func) {
    // Find the index of the first element that satisfies the condition
    const index = arr.findIndex(func);
    console.log("indes", index);
  
    // If an element satisfying the condition is found, return the remaining elements
    if (index !== -1) {
      console.log(arr.slice("slice", index));
      return arr.slice(index);
    }
  
    // If no element satisfies the condition, return an empty array
    return [];
  }
  
  // Test cases
  console.log(dropElements([1, 2, 3, 4], function(n) {return n >= 3;})); // [3, 4]

// const originalArray = [1, 2, 3, 4, 5];

// // Using slice with a single parameter to extract elements from an index to the end
// const slicedArray = originalArray.slice(2);

// console.log(slicedArray); // Output: [3, 4, 5]



// const originalArray = [1, 2, 3, 4, 5];

// // Using slice to extract a portion of the array
// const slicedArray = originalArray.slice(1, 4);

// console.log(slicedArray); // Output: [2, 3, 4]


// const result = dropElements([1, 2, 3, 4], function(n) {
//   return n >= 3;
// });

// console.log(result); // Output: [3, 4]


