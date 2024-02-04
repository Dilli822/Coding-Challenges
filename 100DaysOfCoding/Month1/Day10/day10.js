const GM = 398600.4418; // Gravitational constant times Earth's mass
const earthRadius = 6367.4447; // Earth's radius in kilometers

function orbitalPeriod(arr) {
  return arr.map(body => {
    // a = earthRadius+avgAlt
    const semiMajorAxis = earthRadius + body.avgAlt;
    const period = Math.round(2 * Math.PI * Math.sqrt((semiMajorAxis ** 3) / GM));

    return { name: body.name, orbitalPeriod: period };
  });
}

// Example usage
console.log(orbitalPeriod([{ name: "sputnik", avgAlt: 35873.5553 }])); 
// Output: [{name: "sputnik", orbitalPeriod: 86400}]

console.log(orbitalPeriod([
  { name: "iss", avgAlt: 413.6 },
  { name: "hubble", avgAlt: 556.7 },
  { name: "moon", avgAlt: 378632.553 }
])); 
// Output: [{name : "iss", orbitalPeriod: 5557}, {name: "hubble", orbitalPeriod: 5734}, {name: "moon", orbitalPeriod: 2377399}]




