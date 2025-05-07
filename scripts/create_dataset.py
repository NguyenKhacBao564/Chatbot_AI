import json
import random
import calendar
import os

# Dataset gốc
original_data = [
    {"query": "Tôi muốn đi tour Đà Lạt", "intent": "find_tour_with_location"},
    {"query": "Cho tôi thông tin tour Đà Lạt", "intent": "find_tour_with_location"},
    {"query": "Tôi cần đặt tour ở Đà Lạt", "intent": "find_tour_with_location"},
    {"query": "Có tour nào ở Đà Lạt không?", "intent": "find_tour_with_location"},
    {"query": "Tôi muốn khám phá Đà Lạt qua tour", "intent": "find_tour_with_location"},
    {"query": "Tìm tour Đà Lạt cho tôi", "intent": "find_tour_with_location"},
    {"query": "Tôi muốn đặt tour Đà Lạt tháng 6", "intent": "find_tour_with_location_and_time"},
    {"query": "Tour Đà Lạt khởi hành ngày 12/7", "intent": "find_tour_with_location_and_time"},
    {"query": "Tôi cần tour Đà Lạt vào cuối tuần sau", "intent": "find_tour_with_location_and_time"},
    {"query": "Tour Đà Lạt ngày 1/8 có không?", "intent": "find_tour_with_location_and_time"},
    {"query": "Tôi muốn đi tour Đà Lạt vào dịp lễ 2/9", "intent": "find_tour_with_location_and_time"},
    {"query": "Tìm tour Đà Lạt khởi hành 20/6", "intent": "find_tour_with_location_and_time"},
    {"query": "Tour Đà Lạt khoảng 15/5", "intent": "find_tour_with_location_and_time"},
    {"query": "Tôi muốn tour Đà Lạt vào đầu tháng 7", "intent": "find_tour_with_location_and_time"},
    {"query": "Có tour Đà Lạt nào ngày 30/4 không?", "intent": "find_tour_with_location_and_time"},
    {"query": "Tour Đà Lạt khởi hành tuần đầu tháng 8", "intent": "find_tour_with_location_and_time"},
    {"query": "Tôi muốn tour Đà Lạt giá dưới 2 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Tour Đà Lạt nào rẻ dưới 4 triệu?", "intent": "find_tour_with_location_and_price"},
    {"query": "Tôi cần tour Đà Lạt tầm 1,5 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Tìm tour Đà Lạt giá khoảng 3 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Có tour Đà Lạt nào dưới 5 triệu không?", "intent": "find_tour_with_location_and_price"},
    {"query": "Tôi muốn đi tour Đà Lạt giá rẻ dưới 2,5 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Tour Đà Lạt giá dưới 1 triệu có không?", "intent": "find_tour_with_location_and_price"},
    {"query": "Tôi cần tour Đà Lạt giá khoảng 2 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Tour Đà Lạt nào dưới 3,5 triệu?", "intent": "find_tour_with_location_and_price"},
    {"query": "Tìm tour Đà Lạt giá rẻ dưới 4 triệu", "intent": "find_tour_with_location_and_price"},
    {"query": "Tôi muốn đặt tour đi Đà Lạt", "intent": "find_tour_with_location"},
    {"query": "Có tour Đà Lạt nào vào ngày 10/10 không?", "intent": "find_tour_with_location_and_time"},
    {"query": "Tour Đà Lạt giá dưới 6 triệu có không?", "intent": "find_tour_with_location_and_price"},
    {"query": "Tôi muốn tour Đà Lạt khởi hành ngày 5/5", "intent": "find_tour_with_location_and_time"}
]

# Các danh sách để tạo dữ liệu
locations = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bạc Liêu", "Bắc Giang", "Bắc Kạn", "Bắc Ninh",
    "Bến Tre", "Bình Dương", "Bình Định", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cao Bằng", "Cần Thơ", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lạng Sơn", "Lào Cai", "Lâm Đồng", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam",
    "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
    "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP. Hồ Chí Minh", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]

time_expressions = [
    {"template": "tháng {month}", "params": ["month"]},
    {"template": "ngày {day}/{month}", "params": ["day", "month"]},
    {"template": "ngày {day} tháng {month}", "params": ["day", "month"]},
    {"template": "cuối tuần sau", "params": []},
    {"template": "khoảng {day}/{month}", "params": ["day", "month"]},
    {"template": "vào đầu tháng {month}", "params": ["month"]},
    {"template": "tuần đầu tháng {month}", "params": ["month"]},
    {"template": "vào ngày {day}/{month}", "params": ["day", "month"]},
    {"template": "từ ngày {day}/{month}", "params": ["day", "month"]},
    {"template": "đầu tháng {month}", "params": ["month"]},
    {"template": "giữa tháng {month}", "params": ["month"]},
    {"template": "cuối tháng {month}", "params": ["month"]},
    {"template": "trong tháng {month}", "params": ["month"]},
    {"template": "trong khoảng {day}/{month}", "params": ["day", "month"]},
    {"template": "vào dịp hè tháng {month}", "params": ["month"]},
    {"template": "vào thời điểm khoảng {day}/{month}", "params": ["day", "month"]},
    {"template": "tháng {month} năm nay", "params": ["month"]},
    {"template": "vào kỳ nghỉ hè", "params": []},
    {"template": "dịp lễ tới", "params": []}
]

price_expressions = [
    "dưới {price} triệu",
    "khoảng {price} triệu",
    "tầm {price} triệu",
    "giá rẻ dưới {price} triệu",
    "giá dưới {price} triệu",
    "ngân sách khoảng {price} triệu",
    "tài chính tầm {price} triệu",
    "tour rẻ hơn {price} triệu",
    "trong mức giá {price} triệu",
    "chi phí khoảng {price} triệu",
    "tour khoảng giá {price} triệu",
    "chi khoảng {price} triệu",
    "tour giá tầm {price} triệu",
    "mức giá dưới {price} triệu",
    "dưới mức {price} triệu",
    "tối đa {price} triệu",
    "không quá {price} triệu",
    "giá khoảng dưới {price} triệu",
    "trong khoảng dưới {price} triệu"
]

query_templates_location = [
    "Tôi muốn đi tour {location}",
    "Cho tôi thông tin tour {location}",
    "Tôi cần đặt tour ở {location}",
    "Có tour nào ở {location} không?",
    "Tôi muốn khám phá {location} qua tour",
    "Tìm tour {location} cho tôi",
    "Tôi muốn đặt tour đi {location}",
    "Có tour nào tại {location} không?",
    "Bạn có tour nào tới {location} không?",
    "Tour du lịch nào đến {location} hiện có?",
    "Tôi quan tâm tour đến {location}",
    "Tôi muốn đăng ký tour đi {location}",
    "Bạn gợi ý tour nào ở {location} không?",
    "Tư vấn tour du lịch tại {location} giúp tôi"
]

query_templates_location_time = [
    "Tôi muốn đặt tour {location} {time}",
    "Tour {location} khởi hành {time}",
    "Tôi cần tour {location} vào {time}",
    "Tour {location} {time} có không?",
    "Tôi muốn đi tour {location} vào {time}",
    "Tìm tour {location} khởi hành {time}",
    "Tour {location} khoảng {time}",
    "Tôi muốn tour {location} vào {time}",
    "Có tour {location} nào {time} không?",
    "Tour {location} khởi hành {time}",
    "Tôi muốn đi {location} vào {time}, có tour không?",
    "Bạn có tour nào đi {location} vào {time} không?",
    "Có tour nào đến {location} khởi hành {time} không?",
    "Tour nào tới {location} {time} bạn gợi ý?",
    "Tôi định đi {location} vào {time}, tour nào phù hợp?"
]

query_templates_location_price = [
    "Tôi muốn tour {location} {price}",
    "Tour {location} nào rẻ {price}?",
    "Tôi cần tour {location} {price}",
    "Tìm tour {location} {price}",
    "Có tour {location} nào {price} không?",
    "Tôi muốn đi tour {location} {price}",
    "Tour {location} {price} có không?",
    "Tôi cần tour {location} {price}",
    "Tour {location} nào {price}?",
    "Tìm tour {location} {price}",
    "Tôi đang tìm tour {location} {price}",
    "Bạn có tour nào đến {location} {price} không?",
    "Tour đi {location} {price} có gì không?",
    "Gợi ý tour {location} khoảng {price} giúp tôi",
    "Tôi muốn đi {location} với chi phí {price}"
]

query_templates_out_of_scope = [
    "Hôm nay thời tiết thế nào?",
    "Ai là tổng thống Việt Nam?",
    "Bạn tên gì?",
    "Làm sao để nấu phở?",
    "Tôi muốn học tiếng Anh",
    "Máy tính của tôi bị hư",
    "Bạn có thể kể chuyện không?",
    "Tôi đang buồn quá",
    "Giúp tôi giải toán với",
    "Bạn có biết chơi cờ vua không?",
    "Tôi muốn đi du lịch nước ngoài",
    "Có gì hay ở Đà Nẵng không?",
    "Tôi muốn đặt mua điện thoại",
    "Làm sao để sửa xe máy?",
    "Tôi bị mất điện thoại rồi",
    "Bạn có biết chơi guitar không?",
    "Bạn có thể kể một câu chuyện cười không?",
    "Làm sao để giảm cân hiệu quả?",
    "Bài hát đang nổi hiện nay là gì?",
    "Con mèo của tôi bị ốm",
    "Trái đất quay quanh mặt trời bao lâu?",
    "Tôi muốn học lập trình Python",
    "Bạn có thể nói tiếng Anh không?",
    "Hôm nay có phim gì hay không?",
    "Cách trồng cây cà chua như thế nào?",
    "Tôi đang tìm việc làm",
    "Nên đầu tư vào cổ phiếu nào?",
    "Lịch sử Việt Nam bắt đầu từ khi nào?",
    "Tôi bị đau đầu, phải làm sao?",
    "Bạn có thể làm thơ không?",
    "Cho tôi công thức nấu bánh mì",
    "Tôi muốn nghe nhạc",
    "Tôi đang buồn, bạn giúp tôi được không?",
    "Làm sao để thi đậu đại học?",
    "Bạn có biết ai là ca sĩ nổi tiếng nhất không?",
    "Hãy nói cho tôi một bí mật",
    "Thời gian bay từ Hà Nội đến Tokyo là bao lâu?",
    "Cách chơi Liên Quân Mobile là gì?",
    "Số pi có bao nhiêu chữ số?",
    "Bạn có thể gọi pizza giúp tôi không?",
    "Sài Gòn có mưa không hôm nay?",
    "Bạn nghĩ gì về trí tuệ nhân tạo?"
]


# Tạo dữ liệu mới
new_data = []

# Tạo 259 mẫu cho find_tour_with_location
for _ in range(259):
    location = random.choice(locations)
    template = random.choice(query_templates_location)
    query = template.format(location=location)
    new_data.append({"query": query, "intent": "find_tour_with_location"})

# Tạo 323 mẫu cho find_tour_with_location_and_time
for _ in range(323):
    location = random.choice(locations)
    time_expr = random.choice(time_expressions)
    template = random.choice(query_templates_location_time)
    
    params = {}
    if "month" in time_expr["params"]:
        params["month"] = random.randint(1, 12)
    if "day" in time_expr["params"]:
        month = params.get("month", random.randint(1, 12))
        _, last_day = calendar.monthrange(2025, month)  # Kiểm tra số ngày trong tháng
        params["day"] = random.randint(1, last_day)
    
    time = time_expr["template"].format(**params)
    query = template.format(location=location, time=time)
    new_data.append({"query": query, "intent": "find_tour_with_location_and_time"})

# Tạo 356 mẫu cho find_tour_with_location_and_price
for _ in range(356):
    location = random.choice(locations)
    template = random.choice(query_templates_location_price)
    price_template = random.choice(price_expressions)
    price = random.uniform(1, 6)
    price = round(price, 1)
    price_str = price_template.format(price=price)
    query = template.format(location=location, price=price_str)
    new_data.append({"query": query, "intent": "find_tour_with_location_and_price"})

for q in query_templates_out_of_scope:
    new_data.append({"query": q, "intent": "out_of_scope"})


# Kết hợp dữ liệu gốc và dữ liệu mới
extended_data = original_data + new_data


try:
    # os.makedirs(os.path.dirname("extended_intent_train_data.json"), exist_ok=True)
    with open("extended_intent_train_data.json", "w", encoding="utf-8") as f:
        json.dump(extended_data, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo dataset mới với {len(extended_data)} mẫu")
except Exception as e:
    print(f"Lỗi khi lưu file: {e}")