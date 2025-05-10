import re
from google_genAI import get_genai_response

def extract_price_vn(query):
    result = get_genai_response("Trích xuất giá tiền từ câu sau: '" + query +"'. Trả về giá tiền dạng x.xxx.xxx chứ không thêm bất cứ phản hồi nào cả, nếu không trích được giá thì trả về None")
    print("Result: ", result)
    if "None" in result:
        return "None"
    return result
    # Biểu thức chính quy mở rộng để xử lý các định dạng giá
    # Bao gồm: "5 triệu", "200 nghìn", "45.5 triệu", "500k", "1tr5", "1.5tr", "2 tỷ"
    # query = re.sub(r'\b\d{1,2}\s*(ngày|tháng|năm)\b', '', query, flags=re.IGNORECASE)
    # pattern = r'(\d+[.,]?\d*)\s*(triệu|nghìn|đồng|k|m|tr(?:ieu)?|tỷ)(?:\s*đồng)?(?:\s*(\d+))?'
    
    # # Tìm tất cả các giá trị khớp trong query
    # matches = re.findall(pattern, query, re.IGNORECASE)
    
    # prices = []
    # for match in matches:
    #     amount, unit, extra = match
    #     # Chuyển đổi số về dạng float
    #     amount = float(amount.replace(',', '.'))
        
    #     # Xử lý trường hợp như "1tr5" (1.5 triệu)
    #     if extra and unit.lower() in ['tr', 'm', 'triệu']:
    #         amount += float(extra) / 10  # Thêm phần thập phân từ extra (ví dụ: "5" -> 0.5)
        
    #     # Chuẩn hóa đơn vị
    #     if unit.lower() in ['triệu', 'm', 'tr']:
    #         amount *= 1_000_000
    #     elif unit.lower() in ['nghìn', 'k']:
    #         amount *= 1_000
    #     elif unit.lower() == 'tỷ':
    #         amount *= 1_000_000_000
    #     elif unit.lower() == 'đồng':
    #         pass  # Đã là đồng
        
    #     prices.append(int(amount))
    
    # # result = get_genai_response("Trích xuất giá tiền từ câu hỏi: trả về giá tiền dạng x.xxx.xxx chứ không thêm bất cứ phản hồi nào cả, nếu không trích được giá thì trả về None" + query)

    # return prices[0] if prices else "None"

# Hàm chatbot
# def chatbot_response(user_input):
#     prices = extract_price_vn(user_input)
#     print("Giá tiền được trích xuất:", prices)
    # print("isdigit: ", [str(price).isdigit() for price in prices])
    # if prices:
    #     return f"Giá tiền được trích xuất: {', '.join([f'{price:,} đồng' for price in prices])}"
    # else:
    #     return "Không tìm thấy thông tin giá tiền trong câu của bạn."

# # Test với các trường hợp mở rộng
# queries = [
#     "tôi muốn đi Phú yên vào ngày 22 tháng 9 Mua điện thoại giá 5 triệu đồng",
#     "Cái áo này 200 nghìn",
#     "Xe máy giá 45.5 triệu",
#     "Nhà giá 2 tỷ đồng",
#     "Laptop 500k",
#     "Điện thoại 1tr5",
#     "Máy tính khoảng 2 triệu rưỡi",
#     "Không có giá"
# ]

# for query in queries:
#     print(f"Query: {query}")
#     print(f"Response: {extract_price_vn(query)}\n")