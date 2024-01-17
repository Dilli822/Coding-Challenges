function rot13(str) {
    // Iterate through each character in the string
    return str.split('').map(function(char) {
      // Check if the character is a letter
      if (char.match(/[A-Za-z]/)) {
        // Determine if it's uppercase or lowercase and perform ROT13 transformation
        const charCode = char.charCodeAt(0);
        const base = char >= 'a' ? 'a'.charCodeAt(0) : 'A'.charCodeAt(0);
        return String.fromCharCode((charCode - base + 13) % 26 + base);
      } else {
        // If it's not a letter, leave it unchanged
        return char;
      }
    }).join('');
  }
  
  // Example usage
  console.log(rot13("SERR PBQR PNZC")); // Output: "FREE CODE CAMP"
  
  
  
  // const charCode = 90;
  // const base = 65;
  
  // // ROT13 transformation: (charCode - base + 13) % 26 + base
  // const transformedCharCode = (charCode - base + 13) % 26 + base;
  
  // // ROT13 decryption: (charCode - base - 13 + 26) % 26 + base
  // const decryptedCharCode = (charCode - base - 13 + 26) % 26 + base;
  
  // console.log("Original character code:", charCode);
  // console.log("Transformed character code:", transformedCharCode);
  // console.log("Decrypted character code:", decryptedCharCode);
  
  
  // Here's the breakdown:
  
  // charCode: 90 (Unicode value of 'Z')
  // base: 65 (Unicode value of 'A')
  // ROT13 Transformation:
  // (90 - 65 + 13) % 26 + 65
  // 38 % 26 + 65
  // 12 + 65
  // Transformed character code: 77 (Unicode value of 'M')
  // ROT13 Decryption:
  // (90 - 65 - 13 + 26) % 26 + 65
  // 8 % 26 + 65
  // 8 + 65
  // Decrypted character code: 73 (Unicode value of 'I')
  // So, if the original character code is 90 ('Z') with a base of 65 ('A'), the ROT13 transformation results in the character 'M', and the decryption brings it back to the original character 'I'.




  /*
  function rot13(str) {
  // this is a kind of encryption method in js 
  // where if character a or A comes then it is shifted to the 13th forward from for example if it is A then it is N
  // so if the given string is A then in cipher text it is N
  // so similar we take each character and make it frward 13th times and decode or convert it using loop or any iterations
  // SERR PBQR PNZC is actually FREE CODE CAMP
  return str;
}

rot13("SERR PBQR PNZC");
  */