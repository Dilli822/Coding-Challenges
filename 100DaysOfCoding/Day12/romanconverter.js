// function convertToRoman(num) {
//   // create a array of objectst having roman numeral values
//   // identify if the number of digits used in it
//   // make roman numerals as value instead of symbols
//   // suppose if it is 1 digit for example if it is 1 then we have for 1 number roman numerals = I
//   // that means if we get param as 3 since, we have no direct numerals for number 3 we look for the highest numerals for number 3 since higher numerals for 3 is 4 but 4a> 3. so we look lower numerals which is 1 or I so we add the I 3 times which is III as we have param 3
 
//  return num;
// }
// convertToRoman(36);

// // for36 we have 40 which XL IN ROMAN NUMERALS IS HIGHEST
// // SINCE 40 > 36 WE NEED TO FIND THE LOWER NUMERALS WHICH IS X = 10 

// Now, let's go through the iterations:

// Iteration 1:
// currentNumeral: { value: 10, numeral: 'X' }
// Since num (36) is greater than or equal to 10, we append 'X' to the result and subtract 10 from num (now num is 26).
// Iteration 2:
// currentNumeral: { value: 10, numeral: 'X' }
// Again, num (26) is greater than or equal to 10, so we append another 'X' to the result and subtract 10 from num (now num is 16).
// Iteration 3:
// currentNumeral: { value: 10, numeral: 'X' }
// Once more, num (16) is greater than or equal to 10, so we append one more 'X' to the result and subtract 10 from num (now num is 6).
// Iteration 4:
// currentNumeral: { value: 5, numeral: 'V' }
// Now, num (6) is greater than or equal to 5, so we append 'V' to the result and subtract 5 from num (now num is 1).
// Iteration 5:
// currentNumeral: { value: 1, numeral: 'I' }
// Finally, num (1) is greater than or equal to 1, so we append 'I' three times to the result and subtract 3 from num (now num is 0).
// The final result is 'XXXVI', which is the Roman numeral representation of the input value 36.

function convertToRoman(num) {
    // Define an array of objects with Roman numeral values and their corresponding Arabic values
    const romanNumerals = [
      { value: 1000, numeral: 'M' },
      { value: 900, numeral: 'CM' },
      { value: 500, numeral: 'D' },
      { value: 400, numeral: 'CD' },
      { value: 100, numeral: 'C' },
      { value: 90, numeral: 'XC' },
      { value: 50, numeral: 'L' },
      { value: 40, numeral: 'XL' },
      { value: 10, numeral: 'X' },
      { value: 9, numeral: 'IX' },
      { value: 5, numeral: 'V' },
      { value: 4, numeral: 'IV' },
      { value: 1, numeral: 'I' }
    ];
  
    let result = '';
  
    // Loop through the Roman numeral array
    for (let i = 0; i < romanNumerals.length; i++) {
      const currentNumeral = romanNumerals[i];
  
      // Check if the current numeral's value is less than or equal to the input number
      while (num >= currentNumeral.value) {
        // Add the Roman numeral to the result
        result += currentNumeral.numeral;
        // Subtract the value of the Roman numeral from the input number
        num -= currentNumeral.value;
      }
    }
  
    return result;
  }
  
  console.log(convertToRoman(36)); // Output: XXXVI
  
  
  
  // 1. Create an array of objects to store Roman numeral values and their corresponding Arabic values. Each object should have a 'value' and 'numeral' property. Arrange the array in descending order of 'value'.
  // 2. Initialize an empty string (result) to store the final Roman numeral.
  // 3. Iterate through the array of Roman numerals:
  //     i. For each numeral, check if its value is less than or equal to the input number.
  //     ii. If true, append the numeral to the result and subtract its value from the input number.
  //     iii. Repeat this process until the input number becomes 0.
  //4. Return the final result.
  