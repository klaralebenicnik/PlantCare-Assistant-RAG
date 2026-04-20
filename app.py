import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. SETUP (App Title and Icon)
st.set_page_config(page_title="PlantCare Assistant", page_icon="🌿")
st.title("🌿 Indoor Houseplant Care Assistant")
st.subheader("A simple guide for happy plants")
st.divider() 

# 2. DATA: 10 Specific Plant Care Documents
DOCUMENTS = [
    "Watering is one of the most important and commonly misunderstood aspects of houseplant care. Most indoor plants prefer soil that is slightly moist but not waterlogged. Overwatering can lead to root rot, while underwatering causes wilting and dry leaves. A good rule of thumb is to check the top 2-3 cm of soil. If it feels dry, it is usually time to water. Different plants have different needs; for example, succulents and cacti require much less frequent watering than tropical plants like ferns. The type of pot also matters, as terracotta pots dry out faster than plastic ones. Drainage holes are essential to prevent excess water from accumulating at the bottom. Seasonal changes affect watering frequency as well, with plants typically needing less water during winter when growth slows.",
    "The type of soil used for houseplants plays a crucial role in their health. A good potting mix provides support, retains moisture, and allows excess water to drain. Standard indoor potting soil is usually made from peat, coconut coir, perlite, and compost. Different plants require specialized mixes; for example, succulents need a fast-draining cactus mix, while orchids thrive in bark-based substrates. Poor soil can compact over time, reducing aeration and leading to root problems. Adding materials like perlite or sand improves drainage, while compost adds nutrients. Repotting with fresh soil every 1-2 years helps maintain plant health. Choosing the right soil mix ensures roots receive both oxygen and moisture in the correct balance.",
    "Light is essential for photosynthesis, and different houseplants have varying light requirements. Bright, indirect light is ideal for many tropical plants, while direct sunlight can scorch sensitive leaves. Low-light plants like snake plants and pothos can tolerate dim conditions but will grow more slowly. South-facing windows provide the strongest light, while north-facing windows offer the least. Artificial grow lights can supplement natural light, especially in darker homes or during winter months. Signs of insufficient light include leggy growth and pale leaves, while too much light can cause browning or burnt spots. Understanding your plant’s natural habitat helps determine its lighting needs and ensures optimal growth.",
    "Many popular houseplants originate from tropical environments and prefer higher humidity levels than typical indoor air provides. Dry air, especially in winter due to heating, can cause brown leaf tips and slow growth. Increasing humidity can be done by misting plants, using a humidifier, or placing plants on pebble trays filled with water. Grouping plants together also helps create a more humid microclimate. Some plants, like ferns and calatheas, are particularly sensitive to low humidity, while others, like succulents, prefer dry conditions. Good air circulation is also important to prevent fungal diseases. Maintaining appropriate humidity levels helps plants thrive and improves their overall appearance.",
    "Houseplants require nutrients such as nitrogen, phosphorus, and potassium to grow. While potting soil provides some nutrients, these are depleted over time, making fertilization necessary. Most plants benefit from feeding during the growing season (spring and summer) and require less or no fertilizer during winter. Liquid fertilizers are commonly used and are typically diluted with water before application. Over-fertilizing can damage roots and cause salt buildup in the soil, leading to leaf burn. Slow-release fertilizers are another option, providing nutrients gradually over time. Choosing the right fertilizer and applying it correctly supports healthy growth and vibrant foliage.",
    "Repotting is necessary when a plant outgrows its container or when the soil becomes depleted. Signs that a plant needs repotting include roots growing out of drainage holes, slow growth, and soil drying out too quickly. When repotting, it is important to choose a pot that is only slightly larger than the current one to avoid overwatering issues. Fresh potting mix should be used to provide nutrients and improve drainage. During repotting, roots can be gently loosened and inspected for signs of rot or damage. Repotting is best done during the growing season to help plants recover quickly. Proper repotting promotes strong root systems and healthier plants.",
    "Spider mites are tiny pests that feed on plant sap, often causing yellow or speckled leaves. They thrive in dry conditions and are commonly found on indoor plants. A key sign of infestation is fine webbing on leaves and stems. Spider mites reproduce quickly, making early detection important. Treatment includes rinsing plants with water, wiping leaves, and using insecticidal soap or neem oil. Increasing humidity can also help prevent infestations. Regular inspection of plants, especially the undersides of leaves, is essential to catch spider mites early and prevent widespread damage.",
    "Thrips are small, slender insects that damage plants by feeding on their tissues, leaving silvery streaks and black specks. Mealybugs, on the other hand, appear as white, cotton-like clusters on stems and leaves. Both pests can weaken plants and spread quickly if not controlled. Treatment methods include isolating affected plants, removing pests manually, and applying insecticidal soap or neem oil. Consistent monitoring is crucial, as these pests can hide in crevices. Maintaining plant health and cleanliness reduces the likelihood of infestations.",
    "Root rot is a common and serious condition caused by overwatering and poor drainage. It occurs when roots sit in waterlogged soil, leading to fungal growth and decay. Symptoms include yellowing leaves, wilting despite moist soil, and a foul smell from the roots. Affected roots appear brown and mushy instead of firm and white. To treat root rot, the plant must be removed from its pot, damaged roots trimmed, and replanted in fresh, well-draining soil. Preventing root rot involves proper watering practices and ensuring pots have adequate drainage. Early intervention is key to saving the plant.",
    "Selecting the right houseplants depends on your environment and lifestyle. Beginners often benefit from low-maintenance plants like pothos, snake plants, or ZZ plants, which tolerate a range of conditions. Factors to consider include available light, humidity, and how often you can water. Pet owners should also check for plant toxicity. Some plants require more attention, such as frequent watering or high humidity, while others are more forgiving. Matching plant needs to your living conditions increases the chances of success and makes plant care more enjoyable."
]

# 3. SETTINGS: User Controls in the Sidebar
st.sidebar.header("⚙️ RAG Configuration")

c_size = st.sidebar.number_input("Chunk Size", min_value=50, max_value=1000, value=500)
c_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=200, value=50)

st.sidebar.divider()
st.sidebar.info("The system uses these values to process the knowledge base in real-time.")


# 4. THE BRAIN
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=c_size, 
    chunk_overlap=c_overlap
)
docs = text_splitter.create_documents(DOCUMENTS)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# THE FIX: This line checks if we have an old database and clears it 
# so the "50" chunks don't haunt the "500" chunks.
import random
unique_id = f"coll_{c_size}_{c_overlap}_{random.randint(0,100)}"
vectorstore = Chroma.from_documents(docs, embeddings, collection_name=unique_id)

# 5. THE SEARCH: 
query = st.text_input("What is your topic of choice?", placeholder="e.g., Light, Water, or Pests")

if query:
    results = vectorstore.similarity_search(query, k=1)
    st.write("### Found Information:")
    st.info(results[0].page_content)

# 6. SYSTEM STATS
st.divider()
st.write(f"**Number of chunks:** {len(docs)}")
