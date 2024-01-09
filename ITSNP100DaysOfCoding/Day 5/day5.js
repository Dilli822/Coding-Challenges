function smallestCommons(arr) {
    // Step 1: Sort the array
    arr.sort(function(a, b) {
      return a - b;
    });
  
    // Step 2: Create a range array oscillates between give ranges
    var range = [];
    for (var i = arr[0]; i <= arr[1]; i++) {
      range.push(i);
    }
  
    // Step 3: Calculate the LCM of two numbers
    function lcm(a, b) {
      return (a * b) / gcd(a, b);
    }
  
    // Step 3.1: Calculate the GCD of two numbers recursive
    function gcd(x, y) {
      return y === 0 ? x : gcd(y, x % y);
    }
  
    // Step 4: Calculate the LCM of all numbers in the range
    return range.reduce(function(acc, curr) {
      return lcm(acc, curr);
    });
  }
  
  // Example usage:
  var result = smallestCommons([1, 5]);
  console.log(result); // Output: 60
  

// Create a range array: [1, 2, 3, 4, 5]
// Initialize with the first element:
// acc = 1 (initial accumulator value)
// curr = 2 (first element in the range)
// Calculate the LCM of 1 and 2:
// LCM(1, 2) = (1 * 2) / GCD(1, 2) = 2 / 1 = 2
// Use the result as the new accumulator:
// acc = 2
// curr = 3 (next element in the range)
// Calculate the LCM of 2 and 3:
// LCM(2, 3) = (2 * 3) / GCD(2, 3) = 6 / 1 = 6
// Use the result as the new accumulator:
// acc = 6
// curr = 4 (next element in the range)
// Calculate the LCM of 6 and 4:
// LCM(6, 4) = (6 * 4) / GCD(6, 4) = 12
// Use the result as the new accumulator:
// acc = 12
// curr = 5 (next element in the range)
// Calculate the LCM of 12 and 5:
// LCM(12, 5) = (12 * 5) / GCD(12, 5) = 60 / 1 = 60
// So, the final result for the range [1, 5] is 60, which is the smallest common multiple.

// Step 1: Calculate the GCD (using the Euclidean algorithm):

// gcd(2, 3)
// Since y (3 in this case) is not equal to 0, we proceed with gcd(3, 2 % 3)
// Finally, gcd(3, 2) is reached, and the function returns 1.
// So, for the example values a = 2 and b = 3:

// GCD(2, 3) = 1
// Step 2: Calculate the LCM using the formula:

// LCM(2, 3) = (2 * 3) / GCD(2, 3) = (6 / 1) = 6
// So, for the example values a = 2 and b = 3:

// LCM(2, 3) = 6
// In this case, the GCD is 1, and the LCM is 6 for the given example.