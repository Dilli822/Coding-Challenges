function telephoneCheck(str) {
    // Define the regular expression for valid US phone numbers
    const phoneRegex = /^(1\s?)?(\(\d{3}\)|\d{3})([\s\-]?)\d{3}([\s\-]?)\d{4}$/;
  
    // Test the input string against the regex
    if (phoneRegex.test(str)) {
      // Check if the phone number contains only numeric characters
      const numericStr = str.replace(/[^\d]/g, '');
      
      // Check if the country code is 1 (if present)
      if (numericStr.length === 11 && numericStr[0] === '1') {
        return true;
      }
      
      // Check if the numeric string has 10 digits
      return numericStr.length === 10;
    }
  
    return false;
  }
  
  // Test cases
  console.log(telephoneCheck("555-555-5555")); // true

// Algorithm
// /^(1\s?)?(\(\d{3}\)|\d{3})([\s\-]?)\d{3}([\s\-]?)\d{4}$/
// ^: Anchors the regex at the start of the string.
// (1\s?)?: This part is optional (?). It matches an optional "1" at the beginning, followed by an optional whitespace character (\s). 
// The entire group is enclosed in parentheses for grouping.
// (\(\d{3}\)|\d{3}): This part matches either:
// \(\d{3}\): Three digits enclosed in parentheses.
// \d{3}: Or, three consecutive digits without parentheses. This part is enclosed in parentheses for grouping.
// ([\s\-]?): Matches an optional whitespace character (\s) or hyphen (-). This is for handling variations in formatting.
// \d{3}: Matches exactly three digits.
// ([\s\-]?): Another optional whitespace character (\s) or hyphen (-).
// \d{4}: Matches exactly four digits.
// $: Anchors the regex at the end of the string.
// This regular expression is designed to match various valid formats of US phone numbers, including those with or without the country code (1),
// with or without parentheses around the area code, and with different possible separators like spaces or hyphens. It helps ensure that the 
// phone number is correctly formatted and does not contain invalid characters in between the digits.

  
  function telephoneCheck(str) {
    // check if it is number from USA 
    // check if it is number first if it is then it should have 
    // 10 length of 5s are accepted whether it has 
    // these formats 555-555-5555
  // (555)555-5555
   // (555) 555-5555
  // 555 555 5555
  // 5555555555
  // 1 555 555 5555
  // or USA nphone number 
  // check if there are any invalid strings or symbols in between the numbers for eg 12&$#3454
  // +1 (212) 555-1212
  // +1 (415) 867-5309
  // +1 (617) 253-1000
  // else return false if there ( "()") and if there pattern like 55 55-55-555-5 that means our. 
    return true;
  }