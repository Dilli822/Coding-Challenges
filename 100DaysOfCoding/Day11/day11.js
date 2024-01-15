// function palindrome(str) {
//   // first check the string from the left most side
//   // then it checks upto the middle
//   // then check the string from the right most side
//   // if both comes true or matched then only return true
//   // that means it doesnot matter whether the strings characters are upper or lower
//   // if above conditions are not meet then return false
//   // while checking the each character we can split the string and use looping for each characters from left to right and right to left
//   return true;
// }

// palindrome("eye");

function palindrome(str) {
    // Remove non-alphanumeric characters and convert to lowercase
    const cleanedStr = str.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
  
    // Compare the string from left to right and right to left
    // divide by 2 is done since we need to check the palindrome till the middle point and it will help to optimize the algorithm
    for (let i = 0; i < cleanedStr.length / 2; i++) {
      if (cleanedStr[i] !== cleanedStr[cleanedStr.length - 1 - i]) {
        return false;
      }
    }
  
    // If the loop completes without returning false, it's a palindrome
    return true;
  }
  
  console.log(palindrome("eye")); // true
  console.log(palindrome("_eye")); // true
  console.log(palindrome("race car")); // true
  console.log(palindrome("not a palindrome")); // false
  console.log(palindrome("A man, a plan, a canal. Panama")); // true
  console.log(palindrome("never odd or even")); // true
  console.log(palindrome("nope")); // false
  
  // For the word "race car," let's go through the steps:
  // The cleaned string after removing non-alphanumeric characters and converting to lowercase is "racecar."
  // The loop iterates up to cleanedStr.length / 2, which is 3 in this case.
  // In the first iteration (i = 0), it compares the first character ("r") with the last character ("r"). They match.
  // In the second iteration (i = 1), it compares the second character ("a") with the second-to-last character ("a"). They match.
  // In the third iteration (i = 2), it compares the third character ("c") with the third-to-last character ("c"). They match.
  // Since all comparisons are successful, the function returns true, indicating that "race car" is a palindrome.
  // odd are radar  and even palindrome noon
  // This approach works well because checking only up to half of the string is sufficient due to the symmetric nature of palindromes. The second half of the string is essentially a reversed version of the first half in a palindrome, so checking beyond the midpoint is unnecessary.