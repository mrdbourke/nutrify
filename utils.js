import { v4 as uuidv4 } from 'https://jspm.dev/uuid';

export { uuidv4 };

// Function to reverse map keys and labels - https://stackoverflow.com/a/56781239/7900723
// {1234: "Apple"} -> {"Apple": 1234}
export function reverse_map(obj) {
    return Object.fromEntries(Object.entries(obj).map(([key, value]) => [value.toLowerCase(), key]));
}
