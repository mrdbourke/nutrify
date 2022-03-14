import { createClient } from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm"

// Connect to Supabase
const supabaseUrl = 'https://nqicivptemixgpdaexsl.supabase.co'
const supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5xaWNpdnB0ZW1peGdwZGFleHNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NDQ2NTY3MjksImV4cCI6MTk2MDIzMjcyOX0.HdD4QSNHs8_0b8c2viEq4ulIHaPVgZ5dypmcCF-mVMw"
const supabase = createClient(supabaseUrl, supabaseKey);

export { supabase };
