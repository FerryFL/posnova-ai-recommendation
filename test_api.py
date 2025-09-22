import requests
import json
import time

# Test data sesuai schema Anda
test_products = [
    {
        "id": "59dbe264-2563-4fce-8238-69bd8e6e8b59",
        "nama": "Coco jelly",
        "harga": 17000,
        "gambar": "https://example.com/image.jpg",
        "stok": 2,
        "status": True,
        "kategori": {
            "id": "45f84893-a9b5-43ab-b9c5-3edce446bc2c",
            "nama": "Minuman",
            "status": True
        },
        "UMKM": {
            "id": "1c6f89a1-8fdc-43ce-ba77-8268f15537ab",
            "nama": "Kelapa James Seger"
        },
        "ProdukVarian": []
    },
    {
        "id": "product-2",
        "nama": "Es Teh Manis",
        "harga": 5000,
        "gambar": "https://example.com/image2.jpg",
        "stok": 10,
        "status": True,
        "kategori": {
            "id": "45f84893-a9b5-43ab-b9c5-3edce446bc2c",
            "nama": "Minuman",
            "status": True
        },
        "UMKM": {
            "id": "1c6f89a1-8fdc-43ce-ba77-8268f15537ab",
            "nama": "Kelapa James Seger"
        },
        "ProdukVarian": []
    }
]

test_transactions = [
    {
        "id": "trans-1",
        "tanggalTransaksi": "2024-01-01T00:00:00Z",
        "totalHarga": 50000,
        "totalProduk": 3,
        "transaksiItem": [
            {
                "id": "item-1",
                "jumlah": 2,
                "hargaSatuan": 17000,
                "varianNama": None,
                "produkId": "59dbe264-2563-4fce-8238-69bd8e6e8b59",
                "produk": {
                    "nama": "Coco jelly"
                }
            }
        ]
    }
]

test_cart_items = [
    {
        "id": "59dbe264-2563-4fce-8238-69bd8e6e8b59",
        "nama": "Coco jelly",
        "harga": 17000,
        "gambar": "https://example.com/image.jpg",
        "stok": 2,
        "status": True,
        "jumlah": 1,
        "varianId": None,
        "varianNama": None,
        "kategori": {
            "id": "45f84893-a9b5-43ab-b9c5-3edce446bc2c",
            "nama": "Minuman",
            "status": True
        },
        "UMKM": {
            "id": "1c6f89a1-8fdc-43ce-ba77-8268f15537ab",
            "nama": "Kelapa James Seger"
        },
        "ProdukVarian": []
    }
]

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Starting API Tests...\n")
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        return
    
    # Test 2: Train Content-Based Model
    print("2. Training Content-Based Model...")
    try:
        response = requests.post(
            f"{base_url}/train-content",
            json={"products": test_products},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test 3: Train Collaborative Model
    print("3. Training Collaborative Model...")
    try:
        response = requests.post(
            f"{base_url}/train-collaborative",
            json={"transactions": test_transactions},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test 4: Content-Based Recommendations
    print("4. Testing Content-Based Recommendations...")
    try:
        response = requests.post(
            f"{base_url}/recommend-content",
            json={
                "items": test_cart_items,
                "num_recommendations": 3
            },
            timeout=20
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Method: {result.get('method')}")
        print(f"   Total Found: {result.get('total_found')}")
        print(f"   Recommendations: {len(result.get('recommendations', []))}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test 5: Collaborative Recommendations
    print("5. Testing Collaborative Recommendations...")
    try:
        response = requests.post(
            f"{base_url}/recommend-collaborative",
            json={
                "items": test_cart_items,
                "all_products": test_products,
                "num_recommendations": 3
            },
            timeout=20
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Method: {result.get('method')}")
        print(f"   Total Found: {result.get('total_found')}")
        print(f"   Recommendations: {len(result.get('recommendations', []))}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    print("✅ Tests Complete!")

if __name__ == "__main__":
    test_api()