// Day 2
// array is a special kind of variable that can hold multiple values of data types 
// arrays are special type  object in js 
// const array_name = [item1, item2, ...];  is a syntax of array 
// we can also create an array using new keyword but this is rearly used
const cars = new Array("Saab", "Volvo", "BMW");
console.log(cars);

// using const to declare array is good practice
const arr = ["BMW", "Maruti", false, 45, 0.67, undefined, null];

// accessing and modifying the elements inside the array
// array is indexed and starts from the index = 0
console.log(arr[0]);   // BMW
console.log(arr[1]);  //  Maruti
console.log(arr[2]); //  false

// we can have function, arrays, variables of different types in an array 
arr[0] = Date.UTC; // objects

function sorting(){};

arr[1] = sorting(); // function
arr[2] = cars; // array 

// checking the array type with typeof operator
console.log(typeof(arr)); // object

// length properties of an array
console.log(arr.length);

// to access last element of an array we use length of array - 1
console.log(arr[arr.length - 1]);
// adding and removing the array from the first and the last
console.log(arr.pop());
console.log(arr.push("Volvo"));

// An array of objects is simply an array where each element is an object.
const arrayOfObjects = [
    { name: 'John', age: 25, city: 'New York' },
    { name: 'Alice', age: 30, city: 'San Francisco' },
    { name: 'Bob', age: 22, city: 'Seattle' }
  ];
  
  // Use a while loop to print out information
  let i = 0;
  while (i < arrayOfObjects.length) {
    console.log("Name: " + arrayOfObjects[i].name + ", Age: " + arrayOfObjects[i].age + ", City: " + arrayOfObjects[i].city);
    i++;
  }