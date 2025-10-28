#!/usr/bin/env python3
"""
LinkedIn Profile Scraper
Uses Serper API to search for LinkedIn profiles and Groq LLM to extract structured data.
"""

import os
import json
import csv
import time
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import http.client
from datetime import datetime
from groq import Groq

# Load environment variables
load_dotenv()


class CSVHandler:
    """Handle CSV file operations for saving LinkedIn records."""
    
    def __init__(self, csv_filename: str = "LinkedinRecords.csv"):
        """Initialize CSV handler."""
        self.csv_filename = csv_filename
        self.headers = ["Timestamp", "Full Name", "First Name", "Last Name", "Job Title", 
                       "Company", "LinkedIn URL", "Profile Summary", "Search Status", "Match Score"]
    
    def _file_exists(self) -> bool:
        """Check if CSV file exists."""
        return os.path.exists(self.csv_filename)
    
    def _create_file_with_headers(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self._file_exists():
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.headers)
                print(f"✓ Created new CSV file: {self.csv_filename}")
            except Exception as e:
                print(f"✗ Error creating CSV file: {str(e)}")
    
    def save_to_csv(self, result: Dict, search_inputs: Dict) -> bool:
        """
        Save LinkedIn profile result to CSV file.
        
        Args:
            result: Result dictionary from scraper
            search_inputs: Dictionary with search inputs (first_name, last_name, company, title)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create file with headers if it doesn't exist
            self._create_file_with_headers()
            
            # Prepare records to add
            records_to_add = []
            
            if result.get("status") == "found":
                match = result.get("best_match", {})
                records_to_add.append(self._prepare_record(match, search_inputs, result.get("status"), "high"))
            elif result.get("status") == "multiple_matches":
                best_match = result.get("best_match", {})
                if best_match:
                    records_to_add.append(self._prepare_record(best_match, search_inputs, result.get("status"), "high"))
                
                # Add other matches
                for match in result.get("all_matches", [])[:3]:  # Limit to top 3 other matches
                    if match.get("linkedin_url") != best_match.get("linkedin_url"):
                        records_to_add.append(self._prepare_record(match, search_inputs, "multiple_matches", 
                                                                  match.get("match_score", "medium")))
            else:
                # No match found, save search attempt
                records_to_add.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Full Name": f"{search_inputs.get('first_name', '')} {search_inputs.get('last_name', '')}".strip(),
                    "First Name": search_inputs.get('first_name', ''),
                    "Last Name": search_inputs.get('last_name', ''),
                    "Job Title": search_inputs.get('title', ''),
                    "Company": search_inputs.get('company', ''),
                    "LinkedIn URL": "Not Found",
                    "Profile Summary": result.get('message', 'No matches found'),
                    "Search Status": result.get('status', 'not_found'),
                    "Match Score": "N/A"
                })
            
            # Append rows to CSV file
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for record in records_to_add:
                    row = [
                        record.get("Timestamp", ""),
                        record.get("Full Name", ""),
                        record.get("First Name", ""),
                        record.get("Last Name", ""),
                        record.get("Job Title", ""),
                        record.get("Company", ""),
                        record.get("LinkedIn URL", ""),
                        record.get("Profile Summary", "")[:500],  # Limit summary length
                        record.get("Search Status", ""),
                        record.get("Match Score", "")
                    ]
                    writer.writerow(row)
            
            print(f"\n✓ Saved {len(records_to_add)} record(s) to CSV file: '{self.csv_filename}'")
            return True
            
        except Exception as e:
            print(f"✗ Error saving to CSV file: {str(e)}")
            return False
    
    def _prepare_record(self, match: Dict, search_inputs: Dict, status: str, match_score: str) -> Dict:
        """Prepare a record for saving to CSV."""
        full_name = match.get('full_name', '')
        # Split name if not already split
        name_parts = full_name.split() if full_name else [search_inputs.get('first_name', ''), search_inputs.get('last_name', '')]
        first_name = name_parts[0] if len(name_parts) > 0 else search_inputs.get('first_name', '')
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else (search_inputs.get('last_name', '') if len(name_parts) == 1 else '')
        
        return {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Full Name": full_name or f"{search_inputs.get('first_name', '')} {search_inputs.get('last_name', '')}".strip(),
            "First Name": first_name or search_inputs.get('first_name', ''),
            "Last Name": last_name or search_inputs.get('last_name', ''),
            "Job Title": match.get('title', '') or search_inputs.get('title', ''),
            "Company": match.get('company', '') or search_inputs.get('company', ''),
            "LinkedIn URL": match.get('linkedin_url', ''),
            "Profile Summary": match.get('profile_summary', ''),
            "Search Status": status,
            "Match Score": match_score
        }


class LinkedInScraper:
    """Main scraper class for LinkedIn profiles."""
    
    def __init__(self):
        """Initialize the scraper with API keys."""
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Validate Serper API key
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables.\n"
                           "Please set SERPER_API_KEY in your .env file.")
        
        # Remove whitespace from Serper key
        self.serper_api_key = self.serper_api_key.strip()
        
        # Validate Groq API key
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.\n"
                           "Please set GROQ_API_KEY in your .env file.\n"
                           "Get your key from: https://console.groq.com/keys")
        
        # Remove whitespace from Groq key
        self.groq_api_key = self.groq_api_key.strip()
        
        # Check if user mistakenly put model name instead of API key
        if "llama" in self.groq_api_key.lower() or "gpt" in self.groq_api_key.lower() or "/" in self.groq_api_key:
            raise ValueError(f"ERROR: GROQ_API_KEY appears to be a model name, not an API key!\n"
                           f"Current value: {self.groq_api_key}\n"
                           f"Please update your .env file with your actual Groq API key.\n"
                           f"Get your key from: https://console.groq.com/keys\n"
                           f"The API key should be a long string starting with 'gsk_' or similar.")
        
        # Validate Serper API key format (should be a hex string)
        if len(self.serper_api_key) < 20:
            raise ValueError(f"WARNING: SERPER_API_KEY seems too short ({len(self.serper_api_key)} chars).\n"
                           f"Serper API keys are typically longer. Please verify in your .env file.")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            # Use a current supported text generation model
            # Try multiple models in order of preference
            self.groq_model = "llama-3.1-8b-instant"  # Default: fast and reliable
            # Fallback options: "openai/gpt-oss-20b", "mixtral-8x7b-32768", "gemma2-9b-it"
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client. Check your GROQ_API_KEY.\n"
                           f"Error: {str(e)}\n"
                           f"Get your key from: https://console.groq.com/keys")
    
    def build_search_query(self, first_name: str, last_name: str, 
                          company: Optional[str] = None) -> str:
        """
        Build LinkedIn search query using ONLY name and company.
        
        Args:
            first_name: Person's first name
            last_name: Person's last name
            company: Company name (required for best results)
            
        Returns:
            Constructed search query string
        """
        # Search with name and company: Handle both full and short names
        # Build flexible query that works with short names and company variations
        query = f'"{first_name} {last_name}"'
        
        if company:
            # Add company to query - Serper will match variations
            query += f' {company}'
        else:
            # Warn if no company provided
            print("⚠ Warning: No company provided. Results may be less accurate.")
            print("   Company name helps match profiles with name variations (nicknames/short names).")
        
        query += ' site:linkedin.com/in/'
        
        return query.strip()
    
    def search_with_serper(self, query: str) -> Dict:
        """
        Search using Serper API.
        
        Args:
            query: Search query string
            
        Returns:
            API response dictionary
        """
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            if res.status != 200:
                raise Exception(f"Serper API error: {res.status} - {data.decode('utf-8')}")
            
            response = json.loads(data.decode("utf-8"))
            conn.close()
            
            return response
            
        except Exception as e:
            print(f"Error searching with Serper: {str(e)}")
            return {"error": str(e)}
    
    def extract_linkedin_results(self, serper_response: Dict) -> List[Dict]:
        """
        Extract LinkedIn URLs and profile data from Serper response.
        
        Args:
            serper_response: Response from Serper API
            
        Returns:
            List of dictionaries with LinkedIn profile information
        """
        results = []
        
        if "error" in serper_response:
            return results
        
        # Extract organic results from Serper
        organic_results = serper_response.get("organic", [])
        
        seen_urls = set()  # Avoid duplicates
        
        for item in organic_results:
            link = item.get("link", "")
            # Check for LinkedIn profile patterns
            if "linkedin.com/in/" in link or "linkedin.com/pub/" in link:
                # Normalize URL (remove query parameters, fragments)
                normalized_url = link.split('?')[0].split('#')[0]
                
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    
                    results.append({
                        "title": item.get("title", ""),
                        "link": normalized_url,
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0)
                    })
        
        # Sort by position (relevance)
        results.sort(key=lambda x: x.get("position", 999))
        
        return results
    
    def process_with_groq(self, search_results: List[Dict], 
                         first_name: str, last_name: str,
                         company: Optional[str] = None,
                         title: Optional[str] = None) -> Dict:
        """
        Use Groq LLM to extract structured profile data from search results.
        
        Args:
            search_results: List of LinkedIn search results
            first_name: Target first name
            last_name: Target last name
            company: Target company (optional)
            title: Target title (optional)
            
        Returns:
            Structured profile data dictionary
        """
        if not search_results:
            return {
                "status": "no_results",
                "message": "No LinkedIn profiles found in search results"
            }
        
        # Prepare EXACT Serper results data for LLM - pass raw data
        results_text = "\n\n".join([
            f"=== Result {i+1} ===\n"
            f"Title: {r['title']}\n"
            f"URL: {r['link']}\n"
            f"Snippet: {r['snippet']}\n"
            f"Position: {r.get('position', 'N/A')}"
            for i, r in enumerate(search_results[:10])  # Limit to top 10 results
        ])
        
        prompt = f"""Hey! I need your help finding the right LinkedIn profiles from these search results. The person we're looking for might have used a short name or nickname, and the company name might be shortened too.

Who we're looking for:
- Name: {first_name + " " + last_name if last_name else first_name}
- Company: {company if company else 'Not specified'}

Keep in mind: People often use short names like "nick" instead of "Nicholas", "tarun" instead of "Tharun", or "prabha" instead of "Prabakaran". Company names might also be shortened - someone might search for "AISSS" when the actual company is "AISSS Business" or "AISSS Solutions Inc".

Here are the LinkedIn profiles that came back from the search:
{results_text}

How to find the best matches:

First, focus on the company name - that's usually the most reliable way to find someone. If the person gave us a company name, we should prioritize profiles where:
- The company name matches exactly (like "AISSS" = "AISSS") 
- The company name is part of a longer name (like "AISSS" matches "AISSS Business" or "AISSS Inc")
- The company name appears somewhere in the result (like searching "Tech" should find "Tech Solutions Inc")
- Company name is shortened but matches (searching "AISSS" finds "AISSS Business Solutions" or "AISSS Technologies")

Then, for the name, be flexible because people use nicknames and short forms:
- If they searched "nick", it could be "Nicholas", "Nick", or "Nicky"
- If they searched "tarun", it might be "Tharun" (just a different spelling or variation)
- If they searched "prabha", it could be "Prabakaran" or "Prabakar"
- Don't worry about exact spelling - if the first few letters match and it's in the same company, that's probably the right person
- Consider common name variations and nicknames
- Match partial names - first few letters of first name or last name

The best matches are when:
- Both the company AND name match (even if the name is in a different form)
- Example: searching "nick" at "AISSS" should find "Nicholas Smith at AISSS Business"
- Company matches exactly or partially AND name matches with variations

Good matches are when:
- The company matches and the name is close (maybe first name or last name matches)
- The company matches and the name starts with similar letters
- Company matches partially (like "Tech" in "Tech Solutions Inc") AND name matches

Only include these if there's nothing better:
- Company matches but the name is completely different (only if no better matches exist)
- Name matches but company doesn't (and only if they didn't give us a company to search)

Please give me all the good matches you find, sorted from best to least best. Include at least the top 3-5 profiles that are close matches based on company and name similarity.

Important: Use the exact information from the search results - don't change names or company names. Just pull them exactly as they appear in the Serper results. Extract:
- full_name: exactly as shown in the title
- title: exactly as shown (job title)
- company: exactly as shown (company name)
- linkedin_url: the exact URL from the result
- profile_summary: the exact snippet text

Return your results as JSON like this:

{{
    "status": "found" or "not_found" or "multiple_matches",
    "best_match": {{
        "full_name": "the exact name from the search result title",
        "title": "the exact job title from the result",
        "company": "the exact company name from the result",
        "linkedin_url": "the exact URL from the search result",
        "profile_summary": "the exact snippet text from the result"
    }},
    "all_matches": [
        {{
            "full_name": "exact name from result",
            "title": "exact title from result",
            "company": "exact company from result",
            "linkedin_url": "exact URL",
            "profile_summary": "exact snippet",
            "match_score": "high/medium/low",
            "match_reason": "why this matches, like 'Company matches and name is short form of Nicholas'"
        }}
    ]
}}

Quick examples to help you understand:
- If they searched for "nick" at "AISSS", look for profiles like "Nicholas Smith | Engineer at AISSS Business" - the company matches and nick is short for Nicholas
- If they searched "tarun" at "Tech", find things like "Tharun Kumar | Manager at Tech Solutions Inc" - company matches and tarun is a variation of Tharun
- If they searched "prabha" at "Solutions", match "Prabakaran R | Director at Solutions Inc" - company matches and prabha is short for Prabakaran

Remember: Company is the most important, then be smart about name variations. Give me the closest matches you can find!"""
        
        try:
            # Configure model parameters based on model type
            model_params = {
                "model": self.groq_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant. Always return valid JSON only, no markdown formatting, no code blocks."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1
            }
            
            # Adjust parameters based on model type
            if "gpt-oss" in self.groq_model.lower() or "openai" in self.groq_model.lower():
                # OpenAI OSS models use different parameters
                model_params["max_completion_tokens"] = 4096
                model_params["top_p"] = 1
                model_params["reasoning_effort"] = "medium"
            elif "llama-guard" in self.groq_model.lower():
                # Content moderation models have limits
                model_params["max_tokens"] = 1024
                print("⚠ Warning: Using content moderation model. Consider switching to a text generation model.")
            else:
                # Standard models (llama, mixtral, etc.)
                model_params["max_tokens"] = 2000
            
            response = self.groq_client.chat.completions.create(**model_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Extract JSON from response
            try:
                # Try to find JSON in the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    parsed_data = json.loads(json_text)
                    
                    # Validate structure
                    if "status" not in parsed_data:
                        parsed_data["status"] = "found" if parsed_data.get("best_match") else "not_found"
                    
                    return parsed_data
                else:
                    return json.loads(response_text)
            except json.JSONDecodeError as je:
                # Fallback: return raw data with error
                print(f"JSON parsing error: {str(je)}")
                print(f"Response text: {response_text[:500]}")
                return {
                    "status": "parsing_error",
                    "raw_response": response_text[:1000],
                    "message": f"Failed to parse LLM response as JSON: {str(je)}"
                }
                
        except Exception as e:
            error_msg = str(e)
            # Check if it's a model compatibility issue
            if "model" in error_msg.lower() or "not found" in error_msg.lower() or "max_tokens" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"Model compatibility error: {error_msg}\n"
                               f"Current model: {self.groq_model}\n"
                               f"Recommended: Try changing groq_model to 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', or 'gemma2-9b-it' in the code."
                }
            return {
                "status": "error",
                "message": f"Groq API error: {error_msg}"
            }
    
    def scrape_profile(self, first_name: str, last_name: str,
                      company: Optional[str] = None,
                      title: Optional[str] = None) -> Dict:
        """
        Main method to scrape a LinkedIn profile - simplified approach.
        
        Args:
            first_name: Person's first name
            last_name: Person's last name
            company: Optional company name
            
        Returns:
            Structured profile data
        """
        print(f"\n{'='*60}")
        print(f"Searching LinkedIn: {first_name} {last_name}")
        if company:
            print(f"Company: {company}")
        print(f"{'='*60}\n")
        
        # Build simple search query
        query = self.build_search_query(first_name, last_name, company)
        print(f"Search Query: {query}\n")
        
        # Search with Serper
        serper_response = self.search_with_serper(query)
        
        if "error" in serper_response:
            return {
                "status": "error",
                "message": f"Serper API error: {serper_response['error']}"
            }
        
        # Extract LinkedIn results from Serper
        linkedin_results = self.extract_linkedin_results(serper_response)
        
        if not linkedin_results:
            return {
                "status": "not_found",
                "message": "No LinkedIn profiles found in search results"
            }
        
        print(f"✓ Found {len(linkedin_results)} LinkedIn profile(s) from Serper\n")
        for idx, result in enumerate(linkedin_results[:5], 1):
            print(f"  {idx}. {result.get('title', 'N/A')[:70]}...")
            print(f"     URL: {result.get('link', 'N/A')}\n")
        
        # Use Groq LLM to extract EXACT matching data from Serper
        print("Extracting exact matching data from Serper results using Groq LLM...\n")
        processed_data = self.process_with_groq(
            linkedin_results, first_name, last_name, company, None  # Only use name and company, ignore title
        )
        
        return processed_data
    


def get_user_input() -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Collect user input via CLI.
    
    Returns:
        Tuple of (first_name, last_name, company, title)
    """
    print("\n" + "="*60)
    print("LinkedIn Profile Scraper")
    print("="*60 + "\n")
    
    first_name = input("Enter First Name (required): ").strip()
    if not first_name:
        raise ValueError("First name is required")
    
    last_name = input("Enter Last Name (required): ").strip()
    if not last_name:
        raise ValueError("Last name is required")
    
    company = input("Enter Company (recommended for accurate matching): ").strip()
    company = company if company else None
    
    if not company:
        print("\n⚠ Note: Searching without company may return multiple matches.")
        confirm = input("Continue without company? (y/n): ").strip().lower()
        if confirm != 'y':
            raise ValueError("Search cancelled. Company is recommended for best results.")
    
    # Title is collected but not used in search - kept for compatibility
    title = input("Enter Job Title (optional, not used in search): ").strip()
    title = title if title else None
    
    return first_name, last_name, company, title


def extract_linkedin_urls(result: Dict) -> List[str]:
    """
    Extract LinkedIn profile URLs from result.
    
    Args:
        result: Result dictionary from scraper
        
    Returns:
        List of LinkedIn profile URLs
    """
    linkedin_urls = []
    
    if result.get("status") == "found":
        url = result.get("best_match", {}).get("linkedin_url")
        if url and url != "N/A" and url:
            linkedin_urls.append(url)
    elif result.get("status") == "multiple_matches":
        url = result.get("best_match", {}).get("linkedin_url")
        if url and url != "N/A" and url:
            linkedin_urls.append(url)
        # Add other match URLs
        for match in result.get("all_matches", []):
            url = match.get("linkedin_url")
            if url and url != "N/A" and url not in linkedin_urls:
                linkedin_urls.append(url)
    
    return linkedin_urls


def format_output(result: Dict) -> None:
    """
    Format and display the output with LinkedIn profile URL as primary output.
    
    Args:
        result: Result dictionary from scraper
    """
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")
    
    if result.get("status") == "found":
        match = result.get("best_match", {})
        linkedin_url = match.get('linkedin_url', 'N/A')
        
        # PRIMARY OUTPUT: LinkedIn Profile URL (prominently displayed)
        print("🎯 LINKEDIN PROFILE URL:")
        print("-" * 70)
        if linkedin_url != 'N/A' and linkedin_url:
            print(f"{linkedin_url}\n")
            print("=" * 70)
        else:
            print("N/A\n")
        
        # Additional profile information
        print("\n📋 Profile Details:")
        print("-" * 70)
        print(f"Full Name: {match.get('full_name', 'N/A')}")
        print(f"Title: {match.get('title', 'N/A')}")
        print(f"Company: {match.get('company', 'N/A')}")
        
        profile_summary = match.get('profile_summary', 'N/A')
        if profile_summary and profile_summary != 'N/A':
            print(f"\nProfile Summary:\n{profile_summary}")
        
    elif result.get("status") == "multiple_matches":
        print("⚠ Multiple potential matches found:\n")
        
        matches = result.get("all_matches", [])
        best_match = result.get("best_match")
        
        # Show best match first with LinkedIn URL prominently
        if best_match:
            best_url = best_match.get('linkedin_url', 'N/A')
            print("🎯 RECOMMENDED LINKEDIN PROFILE URL:")
            print("-" * 70)
            if best_url != 'N/A' and best_url:
                print(f"{best_url}\n")
            else:
                print("N/A\n")
            
            print("Best Match Details:")
            print(f"  Name: {best_match.get('full_name', 'N/A')}")
            print(f"  Title: {best_match.get('title', 'N/A')}")
            print(f"  Company: {best_match.get('company', 'N/A')}")
            print("-" * 70)
        
        # Show all other matches
        if len(matches) > 1:
            print(f"\n📋 Other Potential Matches ({len(matches)} total):\n")
            for i, match in enumerate(matches[:5], 1):  # Show top 5
                match_url = match.get('linkedin_url', 'N/A')
                print(f"Match {i}:")
                print(f"  LinkedIn URL: {match_url}")
                print(f"  Name: {match.get('full_name', 'N/A')}")
                print(f"  Title: {match.get('title', 'N/A')}")
                print(f"  Company: {match.get('company', 'N/A')}")
                print(f"  Match Score: {match.get('match_score', 'N/A')}")
                print()
        
    elif result.get("status") == "not_found":
        print("✗ No matching LinkedIn profiles found.")
        if result.get("message"):
            print(f"\n{result['message']}")
    
    else:
        print("✗ Error occurred during search.")
        if result.get("message"):
            print(f"\n{result['message']}")
        if result.get("raw_response"):
            print(f"\nRaw response: {result['raw_response']}")
    
    print("\n" + "="*70)
    
    # Extract and display just the LinkedIn URL(s) for easy copying
    linkedin_urls = extract_linkedin_urls(result)
    
    if linkedin_urls:
        print("\n📎 LinkedIn Profile URL(s) - Copy these:")
        print("-" * 70)
        for url in linkedin_urls:
            print(url)
        print("=" * 70)
    
    # Also output as JSON (optional, can be disabled)
    print("\n📄 Full JSON Output:")
    print(json.dumps(result, indent=2))


def main():
    """Main entry point."""
    import sys
    
    # Check for --url-only flag
    url_only = "--url-only" in sys.argv or "-u" in sys.argv
    
    try:
        # Initialize scraper
        scraper = LinkedInScraper()
        
        # Get user input
        first_name, last_name, company, title = get_user_input()
        
        # Scrape profile
        result = scraper.scrape_profile(first_name, last_name, company, title)
        
        # Display results
        if url_only:
            # Output only LinkedIn URLs (one per line)
            linkedin_urls = extract_linkedin_urls(result)
            if linkedin_urls:
                for url in linkedin_urls:
                    print(url)
            else:
                print("No LinkedIn profile URL found", file=sys.stderr)
                sys.exit(1)
        else:
            format_output(result)
        
        # Save to CSV file
        try:
            csv_handler = CSVHandler("LinkedinRecords.csv")
            search_inputs = {
                "first_name": first_name,
                "last_name": last_name,
                "company": company or "",
                "title": title or ""
            }
            csv_handler.save_to_csv(result, search_inputs)
        except Exception as e:
            print(f"\n⚠ Warning: Failed to save to CSV file: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

