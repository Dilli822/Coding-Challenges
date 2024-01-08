function isPrime(num) {
    (num < 2)? false: true;
      for (let i = 2; i <= Math.sqrt(num); i++) {
          if (num % i === 0) {
              return false;
          }
      }
      return true;
  }
  function sumPrimes(num) {
      let sum = 0;
      for (let i = 2; i <= num; i++) {
          if (isPrime(i)) {
              sum += i;
          }
      }
      return sum;
  }
  const result = sumPrimes(10);
  console.log(result); // Output: 17 (2 + 3 + 5 + 7)
  console.log(sumPrimes(977));
  