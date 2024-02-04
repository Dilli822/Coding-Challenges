function truthCheck(collection, pre) {
    // Check if every object in the collection has a truthy value for the specified property
    return collection.every(item => item[pre]);
  }
  
  // Test the function
  const result = truthCheck(
    [
      { name: "Quincy", role: "Founder", isBot: false },
      { name: "Naomi", role: "", isBot: false },
      { name: "Camperbot", role: "Bot", isBot: true }
    ],
    "isBot"
  );
  
  console.log(result); // Output: false
  
  
  //  a truthy value is a value that is considered true when evaluated in a boolean context. The following values are considered falsy:
  
  // Falsy Values:
  // false
  // 0
  // '' (an empty string)
  // null
  // undefined
  // NaN (Not a Number)
  
  // Truthy Values:
  // Any value that is not falsy is considered truthy. This includes:
  // Non-empty strings ('hello')
  // Numbers other than 0 (42)
  // Objects
  // Arrays
  // Functions
  // true
  // Instances of user-defined classes
  
  
  // Example: Checking if all numbers in an array are even
  // const numbers = [2, 4, 7, 8, 10];
  
  // const allEven = numbers.every(num => num % 2 === 0);
  
  // console.log(allEven); // Output: false
  
  
  // // Example: Checking if all persons in an array are adults
  // const people = [
  //   { name: 'Alice', age: 25 },
  //   { name: 'Bob', age: 30 },
  //   { name: 'Charlie', age: 18 },
  //   { name: 'Diana', age: 22 }
  // ];
  
  // // Check if all persons are adults (age >= 18)
  // const allAdults = people.every(person => person.age >= 18);
  
  // console.log(allAdults); // Output: false
  