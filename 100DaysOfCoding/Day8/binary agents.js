
function binaryAgent(binaryString) {
    // Split the binary string into an array of binary values
    
    const binaryArray = binaryString.split(" ");
    // Convert each binary value to its decimal equivalent, then to a character
    const textArray = binaryArray.map(binary => String.fromCharCode(parseInt(binary, 2)));
  
    // Join the characters to form the final string
    const result = textArray.join("");
  
    return result;
  }
  
  // Test cases
  console.log(binaryAgent("01000001 01110010 01100101 01101110 00100111 01110100 00100000 01100010 01101111 01101110 01100110 01101001 01110010 01100101 01110011 00100000 01100110 01110101 01101110 00100001 00111111"));
  // Output: Aren't bonfires fun!?
  
  console.log(binaryAgent("01001001 00100000 01101100 01101111 01110110 01100101 00100000 01000110 01110010 01100101 01100101 01000011 01101111 01100100 01100101 01000011 01100001 01101101 01110000 00100001"));
  // Output: I love FreeCodeCamp!
  
// const binaryString = "01000001";

// // Convert binary to decimal
// const decimalValue = parseInt(binaryString, 2);

// console.log("Binary String:", binaryString);
// console.log("Decimal Value:", decimalValue);


// const decimalValue = 65;

// // Convert decimal to character
// const character = String.fromCharCode(decimalValue);

// console.log("Decimal Value:", decimalValue);
// console.log("Character:", character);


// const binaryArray = ["01000001", "01110010", "01100101", "01101110", "00100111"];

// // Use map to convert each binary string to its corresponding character
// const textArray = binaryArray.map(binary => String.fromCharCode(parseInt(binary, 2)));

// console.log("Binary Array:", binaryArray);
// console.log("Text Array:", textArray);
