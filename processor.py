from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ZeroShotAgent
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")

# Base Property Features Model
class PropertyFeatures(BaseModel):
    property_type: Optional[str] = Field(None, description="Type of property (house, apartment, condo, etc.)")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    square_footage: Optional[int] = Field(None, description="Square footage of the property")
    year_built: Optional[int] = Field(None, description="Year the property was built")
    address: Optional[str] = Field(None, description="Property address")
    neighborhood: Optional[str] = Field(None, description="Neighborhood or area")
    price: Optional[str] = Field(None, description="Asking price")
    amenities: List[str] = Field(default_factory=list, description="List of amenities")
    parking: Optional[str] = Field(None, description="Parking information")
    lot_size: Optional[str] = Field(None, description="Size of the lot")
    flooring: Optional[str] = Field(None, description="Flooring type")
    heating_cooling: Optional[str] = Field(None, description="Heating and cooling systems")
    exterior: Optional[str] = Field(None, description="Exterior features")
    roof: Optional[str] = Field(None, description="Roof information")
    basement: Optional[str] = Field(None, description="Basement information")
    appliances: List[str] = Field(default_factory=list, description="Included appliances")
    outdoor_features: List[str] = Field(default_factory=list, description="Outdoor features like patio, deck, pool")
    proximity: List[str] = Field(default_factory=list, description="Nearby amenities or locations")
    description: Optional[str] = Field(None, description="General description of the property")

class ResidentialPropertyFeatures(PropertyFeatures):
    schools: Optional[str] = Field(None, description="Nearby schools")
    hoa_fees: Optional[str] = Field(None, description="HOA fees")
    rental_income: Optional[str] = Field(None, description="Potential rental income")
    utilities: Optional[str] = Field(None, description="Utility information")
    interior_features: List[str] = Field(default_factory=list, description="Interior features and finishes")

class CommercialPropertyFeatures(PropertyFeatures):
    zoning: Optional[str] = Field(None, description="Zoning information")
    cap_rate: Optional[str] = Field(None, description="Capitalization rate")
    noi: Optional[str] = Field(None, description="Net operating income")
    lease_terms: Optional[str] = Field(None, description="Lease terms")
    current_tenants: Optional[str] = Field(None, description="Information about current tenants")
    traffic_count: Optional[str] = Field(None, description="Traffic count information")
    usable_area: Optional[int] = Field(None, description="Usable square footage")
    ceiling_height: Optional[str] = Field(None, description="Ceiling height")
    loading_docks: Optional[str] = Field(None, description="Loading dock information")

class LandPropertyFeatures(PropertyFeatures):
    zoning: Optional[str] = Field(None, description="Zoning information")
    topography: Optional[str] = Field(None, description="Land topography")
    soil_type: Optional[str] = Field(None, description="Soil type")
    water_rights: Optional[str] = Field(None, description="Water rights information")
    utilities_access: Optional[str] = Field(None, description="Utilities access")
    road_access: Optional[str] = Field(None, description="Road access information")
    development_potential: Optional[str] = Field(None, description="Development potential")
    environmental_features: List[str] = Field(default_factory=list, description="Environmental features")

class SpecialPurposePropertyFeatures(PropertyFeatures):
    intended_use: Optional[str] = Field(None, description="Intended use of the property")
    current_capacity: Optional[str] = Field(None, description="Current capacity")
    specialized_features: List[str] = Field(default_factory=list, description="Specialized features for intended use")
    licenses: List[str] = Field(default_factory=list, description="Required licenses or permits")
    compliance_info: Optional[str] = Field(None, description="Compliance information")

class ContactInformation(BaseModel):
    name: Optional[str] = Field(None, description="Contact person's name")
    phone: Optional[str] = Field(None, description="Contact phone number")
    email: Optional[str] = Field(None, description="Contact email address")
    company: Optional[str] = Field(None, description="Real estate company name")
    website: Optional[str] = Field(None, description="Website URL")
    office_address: Optional[str] = Field(None, description="Office address")
    available_hours: Optional[str] = Field(None, description="Available hours for contact")

class ImageRequirements(BaseModel):
    exterior: bool = Field(False, description="Exterior image needed")
    living_room: bool = Field(False, description="Living room image needed")
    kitchen: bool = Field(False, description="Kitchen image needed")
    bedrooms: bool = Field(False, description="Bedroom images needed")
    bathrooms: bool = Field(False, description="Bathroom images needed")
    backyard: bool = Field(False, description="Backyard image needed")
    pool: bool = Field(False, description="Pool image needed")
    garage: bool = Field(False, description="Garage image needed")
    basement: bool = Field(False, description="Basement image needed")
    neighborhood: bool = Field(False, description="Neighborhood image needed")
    floor_plan: bool = Field(False, description="Floor plan image needed")
    other_features: List[str] = Field(default_factory=list, description="Other features needing images")

class RealEstateAdData(BaseModel):
    property_type_info: Dict[str, str] = Field(..., description="Type of property information")
    property_features: Union[ResidentialPropertyFeatures, CommercialPropertyFeatures, LandPropertyFeatures, SpecialPurposePropertyFeatures, PropertyFeatures] = Field(..., description="Extracted property features")
    contact_info: ContactInformation = Field(..., description="Extracted contact information")
    image_requirements: ImageRequirements = Field(..., description="Image requirements for the ad")

class RealEstateFeatureExtractor:
    def __init__(self, api_key):
        genai.configure(api_key=gemini_api_key)
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        print("\n\nLLM Done!!\n")
        self.setup_chains()
        print("\n\nSetup_chains Done!!\n")
    
    def setup_chains(self):
        # Property type determination chain template
        property_type_template = """
        You are a real estate expert who can identify the type of property from a description.
        Based on the raw input, determine the specific category and subcategory of the property.
        
        Raw user input: {user_input}
        
        Categorize the property into one of the following main categories and subcategories:
        1. Residential: single-family home, condo/apartment, townhouse, luxury estate, vacation home/rental, mobile/manufactured home
        2. Commercial: office space, retail store, restaurant, hotel/motel, industrial warehouse, mixed-use building
        3. Land: vacant lot, agricultural land, ranch, development land, waterfront land
        4. Special Purpose: senior living, student housing, self-storage, medical office, educational property
        
        Return a JSON object WITHOUT ANY ADDITIONAL TEXT OR EXPLANATIONS with the following structure:
        {{
          "main_category": "one of: Residential, Commercial, Land, Special Purpose",
          "sub_category": "the specific subcategory from above",
          "property_description": "brief 1-2 sentence description of what makes this property fit that category"
        }}
        
        DO NOT include any markdown formatting, code blocks, or explanatory text.
        Return ONLY the JSON object.
        """
        
        property_type_prompt = PromptTemplate(
            input_variables=["user_input"],
            template=property_type_template
        )
        
        self.property_type_chain = LLMChain(
            llm=self.llm,
            prompt=property_type_prompt,
            output_key="property_type_data"
        )
        
        # Enhanced input prompt chain
        enhance_template = """
        You are a real estate expert who can organize and structure raw property descriptions.
        Take the raw user input about a property listing and enhance it by:
        1. Structuring the information clearly
        2. Separating property features from contact information
        3. Making sure all details are preserved
        4. Not adding any information that isn't in the original input
        
        Raw user input: {user_input}
        Property Type: {property_type}
        Sub-Category: {sub_category}
        
        Return the enhanced, well-structured property description:
        """
        
        enhance_prompt = PromptTemplate(
            input_variables=["user_input", "property_type", "sub_category"],
            template=enhance_template
        )
        
        self.enhance_chain = LLMChain(
            llm=self.llm,
            prompt=enhance_prompt,
            output_key="enhanced_input"
        )
        
        # Residential feature extraction prompt
        residential_feature_template = """
        Extract only the residential property features that are explicitly mentioned in the enhanced input.
        Do not infer or assume features that are not clearly stated.
        If a feature is not mentioned, leave it as null or empty list.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following information in a JSON format:
        - Property type (house, apartment, etc.)
        - Number of bedrooms
        - Number of bathrooms
        - Square footage
        - Year built
        - Address
        - Neighborhood/area
        - Price
        - Amenities (as a list)
        - Parking information
        - Lot size
        - Flooring type
        - Heating/cooling systems
        - Exterior features
        - Roof information
        - Basement details
        - Included appliances (as a list)
        - Outdoor features like patios, decks, pools (as a list)
        - Proximity to amenities or locations (as a list)
        - General description of the property
        - Schools nearby
        - HOA fees
        - Rental income potential
        - Utilities information
        - Interior features and finishes
        
        Only include features explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        residential_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=residential_feature_template
        )
        
        self.residential_feature_chain = LLMChain(
            llm=self.llm,
            prompt=residential_prompt,
            output_key="property_features"
        )
        
        # Commercial feature extraction prompt
        commercial_feature_template = """
        Extract only the commercial property features that are explicitly mentioned in the enhanced input.
        Do not infer or assume features that are not clearly stated.
        If a feature is not mentioned, leave it as null or empty list.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following information in a JSON format:
        - Property type (office, retail, etc.)
        - Square footage
        - Usable area
        - Year built
        - Address
        - Business district/area
        - Price/lease rate
        - Amenities (as a list)
        - Parking information
        - Lot size
        - Flooring type
        - Heating/cooling systems
        - Exterior features
        - Roof information
        - Zoning information
        - Capitalization rate
        - Net operating income
        - Lease terms
        - Current tenants information
        - Traffic count information
        - Ceiling height
        - Loading dock information
        - Proximity to major routes, amenities, or locations (as a list)
        - General description of the property
        
        Only include features explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        commercial_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=commercial_feature_template
        )
        
        self.commercial_feature_chain = LLMChain(
            llm=self.llm,
            prompt=commercial_prompt,
            output_key="property_features"
        )
        
        # Land feature extraction prompt
        land_feature_template = """
        Extract only the land property features that are explicitly mentioned in the enhanced input.
        Do not infer or assume features that are not clearly stated.
        If a feature is not mentioned, leave it as null or empty list.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following information in a JSON format:
        - Property type (vacant lot, agricultural, etc.)
        - Lot size/acreage
        - Address/location
        - Price
        - Zoning information
        - Topography
        - Soil type
        - Water rights
        - Utilities access
        - Road access
        - Development potential
        - Environmental features (as a list)
        - Proximity to amenities or locations (as a list)
        - General description of the property
        
        Only include features explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        land_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=land_feature_template
        )
        
        self.land_feature_chain = LLMChain(
            llm=self.llm,
            prompt=land_prompt,
            output_key="property_features"
        )
        
        # Special purpose feature extraction prompt
        special_purpose_template = """
        Extract only the special purpose property features that are explicitly mentioned in the enhanced input.
        Do not infer or assume features that are not clearly stated.
        If a feature is not mentioned, leave it as null or empty list.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following information in a JSON format:
        - Property type (senior living, student housing, etc.)
        - Square footage
        - Year built
        - Address
        - Area/neighborhood
        - Price
        - Intended use
        - Current capacity
        - Specialized features (as a list)
        - Required licenses or permits (as a list)
        - Compliance information
        - Amenities (as a list)
        - Parking information
        - Lot size
        - Heating/cooling systems
        - Exterior features
        - Proximity to amenities or locations (as a list)
        - General description of the property
        
        Only include features explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        special_purpose_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=special_purpose_template
        )
        
        self.special_purpose_feature_chain = LLMChain(
            llm=self.llm,
            prompt=special_purpose_prompt,
            output_key="property_features"
        )
        
        # Generic feature extraction prompt (fallback)
        feature_template = """
        Extract only the property features that are explicitly mentioned in the enhanced input.
        Do not infer or assume features that are not clearly stated.
        If a feature is not mentioned, leave it as null or empty list.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following information in a JSON format:
        - Property type (if mentioned)
        - Number of bedrooms (if residential)
        - Number of bathrooms (if residential)
        - Square footage
        - Year built
        - Address
        - Neighborhood/area
        - Price
        - Amenities (as a list)
        - Parking information
        - Lot size
        - Flooring type
        - Heating/cooling systems
        - Exterior features
        - Roof information
        - Basement details (if applicable)
        - Included appliances (as a list, if applicable)
        - Outdoor features (as a list)
        - Proximity to amenities or locations (as a list)
        - General description of the property
        
        Only include features explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        feature_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=feature_template
        )
        
        self.feature_chain = LLMChain(
            llm=self.llm,
            prompt=feature_prompt,
            output_key="property_features"
        )
        
        # Contact information extraction prompt
        contact_template = """
        Extract only the contact information that is explicitly mentioned in the enhanced input.
        Do not infer or assume information that is not clearly stated.
        If any contact detail is not mentioned, leave it as null.
        
        Enhanced Input: {enhanced_input}
        
        Extract the following contact information in a JSON format:
        - Contact person's name
        - Phone number
        - Email address
        - Real estate company name
        - Website URL
        - Office address
        - Available hours for contact
        
        Only include contact information explicitly mentioned in the input.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        contact_prompt = PromptTemplate(
            input_variables=["enhanced_input"],
            template=contact_template
        )
        
        self.contact_chain = LLMChain(
            llm=self.llm,
            prompt=contact_prompt,
            output_key="contact_info"
        )
        
        # Image requirements identification prompt
        image_template = """
        Based on the extracted property features and property type, determine which aspects of the property would require images for the ad.
        Only mark 'true' for features that were actually mentioned in the original input.
        
        Property Type: {property_type}
        Sub-Category: {sub_category}
        Property Features: {property_features}
        
        For each of the following categories, return true if the feature is mentioned and should have an image, or false otherwise:
        - Exterior
        - Living room
        - Kitchen
        - Bedrooms
        - Bathrooms
        - Backyard
        - Pool
        - Garage
        - Basement
        - Neighborhood
        - Floor plan
        
        Also include a list of any other specific features mentioned that would benefit from images.
        
        Return the result as valid JSON that can be parsed by Python's json.loads() function:
        """
        
        image_prompt = PromptTemplate(
            input_variables=["property_type", "sub_category", "property_features"],
            template=image_template
        )
        
        self.image_chain = LLMChain(
            llm=self.llm,
            prompt=image_prompt,
            output_key="image_requirements"
        )
        
        # ReAct validation chain to prevent hallucination
        validation_template = """
        You are a validation agent for real estate ad information. Review the extracted features and ensure they were explicitly mentioned in the original user input.
        Remove any features that appear to be hallucinated or inferred without clear evidence from the input.
        
        Original User Input: {user_input}
        Enhanced Input: {enhanced_input}
        Property Type: {property_type}
        Sub-Category: {sub_category}
        Extracted Property Features: {property_features}
        Extracted Contact Information: {contact_info}
        
        For each extracted piece of information, verify it against the original input. If you find any discrepancies or inferences not supported by the text, remove them.
        
        Provide the corrected property features and contact information as a valid JSON object with two keys:
        1. "property_features" - containing the validated property features
        2. "contact_info" - containing the validated contact information
        
        Return only valid JSON that can be parsed by Python's json.loads() function:
        """
        
        validation_prompt = PromptTemplate(
            input_variables=["user_input", "enhanced_input", "property_type", "sub_category", "property_features", "contact_info"],
            template=validation_template
        )
        
        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=validation_prompt,
            output_key="validated_data"
        )
    
    def parse_json_safely(self, json_str):
        """Helper function to safely parse JSON strings"""
        if isinstance(json_str, dict):
            return json_str
        
        # If the input is None or empty, return empty dict
        if not json_str:
            return {}
            
        try:
            # Try to parse as is
            return json.loads(json_str)
        except:
            try:
                # Clean up the string and try again - remove markdown code blocks
                clean_str = re.sub(r'```(json)?|```', '', json_str).strip()
                return json.loads(clean_str)
            except:
                try:
                    # Try to find JSON within text - look for content between curly braces
                    match = re.search(r'({.*})', json_str, re.DOTALL)
                    if match:
                        json_content = match.group(1)
                        return json.loads(json_content)
                    else:
                        # Return empty dict if no JSON-like content found
                        print(f"Failed to parse JSON - no JSON-like content found: {json_str}")
                        return {}
                except:
                    # Return empty dict if all fails
                    print(f"Failed to parse JSON after multiple attempts: {json_str}")
                    return {}
    
    def process_input(self, user_input):
        try:
            # Step 1: Determine property type
            print("Determining property type...")
            property_type_result = self.property_type_chain.invoke({"user_input": user_input})
            property_type_json = property_type_result.get("property_type_data", "{}")
            print(f"Property type result: {property_type_json}")
            
            property_type_data = self.parse_json_safely(property_type_json)
            
            if not property_type_data:
                print("Warning: Could not parse property type data properly. Using default values.")
                property_type_data = {"main_category": "Residential", "sub_category": "", "property_description": ""}
            
            main_category = property_type_data.get("main_category", "Residential")
            sub_category = property_type_data.get("sub_category", "")
            property_description = property_type_data.get("property_description", "")
            
            print(f"Detected property type: {main_category}, {sub_category}")
            
            # Step 2: Enhance the user input with property type context
            print("Enhancing user input...")
            enhanced_result = self.enhance_chain.invoke({
                "user_input": user_input,
                "property_type": main_category,
                "sub_category": sub_category
            })
            enhanced_input = enhanced_result.get("enhanced_input", user_input)
            
            # Step 3: Select appropriate feature chain based on property type
            print(f"Extracting features for {main_category} property...")
            if main_category == "Residential":
                feature_chain = self.residential_feature_chain
            elif main_category == "Commercial":
                feature_chain = self.commercial_feature_chain
            elif main_category == "Land":
                feature_chain = self.land_feature_chain
            elif main_category == "Special Purpose":
                feature_chain = self.special_purpose_feature_chain
            else:
                # Default to generic chain
                feature_chain = self.feature_chain
            
            # Step 4: Extract features and contact information
            feature_result = feature_chain.invoke({"enhanced_input": enhanced_input})
            feature_json = feature_result.get("property_features", "{}")
            print(f"Feature extraction result length: {len(str(feature_json))}")
            
            contact_result = self.contact_chain.invoke({"enhanced_input": enhanced_input})
            contact_json = contact_result.get("contact_info", "{}")
            print(f"Contact info extraction result length: {len(str(contact_json))}")
            
            # Parse the JSON results
            property_features = self.parse_json_safely(feature_json)
            contact_info = self.parse_json_safely(contact_json)
            
            if not property_features:
                print("Warning: Could not parse property features properly.")
            
            if not contact_info:
                print("Warning: Could not parse contact information properly.")
            
            # Step 5: Validate extracted information to prevent hallucination
            print("Validating extracted information...")
            validation_input = {
                "user_input": user_input,
                "enhanced_input": enhanced_input,
                "property_type": main_category,
                "sub_category": sub_category,
                "property_features": json.dumps(property_features),
                "contact_info": json.dumps(contact_info)
            }
            
            validated_result = self.validation_chain.invoke(validation_input)
            validated_json = validated_result.get("validated_data", "{}")
            validated_data = self.parse_json_safely(validated_json)
            
            # Use validated data if available, otherwise fall back to original extraction
            if validated_data and isinstance(validated_data, dict):
                property_features = validated_data.get("property_features", property_features)
                contact_info = validated_data.get("contact_info", contact_info)
            
            # Step 6: Determine image requirements based on features and property type
            print("Determining image requirements...")
            image_input = {
                "property_type": main_category,
                "sub_category": sub_category,
                "property_features": json.dumps(property_features)
            }
            image_result = self.image_chain.invoke(image_input)
            image_json = image_result.get("image_requirements", "{}")
            image_requirements = self.parse_json_safely(image_json)
            
            if not image_requirements:
                print("Warning: Could not parse image requirements properly.")
            
            # Step 7: Compile the final result
            result = {
                "property_type_info": {
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "property_description": property_description
                },
                "property_features": property_features,
                "contact_info": contact_info,
                "image_requirements": image_requirements
            }
            
            print("Processing completed successfully.")
            return result
            
        except Exception as e:
            import traceback
            print(f"Error processing input: {str(e)}")
            print(traceback.format_exc())
            # Return a basic structure with error information
            return {
                "error": str(e),
                "property_type_info": {"main_category": "Unknown", "sub_category": "Unknown"},
                "property_features": {},
                "contact_info": {},
                "image_requirements": {}
            }

# API endpoint function
def process_real_estate_ad(user_input):
    extractor = RealEstateFeatureExtractor(gemini_api_key)
    result = extractor.process_input(user_input)
    return result