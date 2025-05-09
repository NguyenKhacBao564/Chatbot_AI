from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from dateparser.search import search_dates

def extract_time(query):
    # Sử dụng dateparser để tìm ngày trong query
    result = search_dates(query, languages=['vi'])
    if result:
        # Lấy đối tượng datetime đầu tiên
        original_text, date_obj = result[0] # result[0] là tuple (datetime, text)
        return date_obj.strftime("%Y-%m-%d")
    return "None"

# # Test
# queries = [
#     "tôi muốn đặt tour đi đà lạt vào tháng 7",
#     "tôi muốn đặt tour đi xuất phát từ ngày 3 tháng 5",
#     "tôi muốn đi vào 3/5",
#     "tôi muốn đặt tour ngày mai",
#     "tôi muốn đặt tour tuần sau",  # Có thể không nhận diện chính xác
#     "tôi muốn đặt tour tháng sau"  # Có thể không nhận diện chính xác
# ]

# print(search_dates("tôi muốn đặt tour đi xuất phát từ ngày 3 tháng 5", languages=['vi']))
# for q in queries:
#     print(f"Query: {q} -> Thời gian: {extract_time(q)}")
    

# # Từ điển ánh xạ tiếng Việt sang số
# MONTH_MAP = {
#     "một": 1, "hai": 2, "ba": 3, "tư": 4, "năm": 5, "sáu": 6,
#     "bảy": 7, "tám": 8, "chín": 9, "mười": 10, "mười một": 11, "mười hai": 12
# }

# DAY_OF_WEEK_MAP = {
#     "thứ hai": 0, "thứ ba": 1, "thứ tư": 2, "thứ năm": 3,
#     "thứ sáu": 4, "thứ bảy": 5, "chủ nhật": 6
# }

# def parse_relative_time(query, now=None):
#     if now is None:
#         now = datetime.now()

#     # Thời gian tương đối
#     if "ngày mai" in query:
#         return (now + timedelta(days=1)).strftime("%Y-%m-%d")
#     elif "tuần sau" in query:
#         return (now + relativedelta(weeks=1)).strftime("%Y-%m-%d")
#     elif "tháng tới" in query:
#         return (now + relativedelta(months=1)).strftime("%Y-%m")

#     # Xử lý thứ trong tuần
#     for day_name, day_index in DAY_OF_WEEK_MAP.items():
#         if day_name in query:
#             current_dow = now.weekday()
#             days_until = (day_index - current_dow + 7) % 7
#             if "tuần sau" in query:
#                 days_until += 7
#             return (now + timedelta(days=days_until)).strftime("%Y-%m-%d")

#     return None

# def extract_time(query, now=None):
#     if now is None:
#         now = datetime.now()
#     current_year = now.year

#     # Các mẫu regex
#     month_pattern = r"tháng\s*(\d{1,2}|một|hai|ba|tư|năm|sáu|bảy|tám|chín|mười|mười một|mười hai)"
#     day_month_pattern = r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2}|một|hai|ba|tư|năm|sáu|bảy|tám|chín|mười|mười một|mười hai)"
#     date_pattern = r"(\d{1,2})[/-](\d{1,2})"
#     year_pattern = r"năm\s*(\d{4})"

#     # Trích xuất năm nếu có
#     year_match = re.search(year_pattern, query)
#     if year_match:
#         current_year = int(year_match.group(1))

#     # Trích xuất ngày và tháng
#     day_month = re.search(day_month_pattern, query)
#     if day_month:
#         day = int(day_month.group(1))
#         month_str = day_month.group(2).lower()
#         month = MONTH_MAP.get(month_str, int(month_str))
#         try:
#             return datetime(current_year, month, day).strftime("%Y-%m-%d")
#         except ValueError:
#             return None

#     # Trích xuất tháng
#     month = re.search(month_pattern, query)
#     if month:
#         month_str = month.group(1).lower()
#         month_num = MONTH_MAP.get(month_str, int(month_str))
#         try:
#             return datetime(current_year, month_num, 1).strftime("%Y-%m")
#         except ValueError:
#             return None

#     # Trích xuất định dạng ngày/tháng (3/5)
#     date = re.search(date_pattern, query)
#     if date:
#         day = int(date.group(1))
#         month = int(date.group(2))
#         try:
#             return datetime(current_year, month, day).strftime("%Y-%m-%d")
#         except ValueError:
#             return None

#     return None

# def extract_all_times(query):
#     now = datetime.now()

#     # Thử với thời gian cụ thể
#     result = extract_time(query, now)
#     if result:
#         return result

#     # Thử với thời gian tương đối
#     result = parse_relative_time(query, now)
#     if result:
#         return result

#     return None

# # Test
# queries = [
#     "tôi muốn đặt tour vào tháng 3",
#     "tôi muốn đặt tour vào tháng ba năm 2026",
#     "tôi muốn đặt tour đi xuất phát từ ngày 3 tháng 5",
#     "tôi muốn đi vào 3/5",
#     "tôi muốn đặt tour tuần sau",
#     "tôi muốn đặt tour tháng tới",
#     "tôi muốn đặt tour ngày mai",
#     "tôi muốn đặt tour thứ tư tuần sau"
# ]

# for q in queries:
#     print(f"Query: {q} -> Thời gian: {extract_all_times(q)}")


# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import re

# def parse_relative_time(query):
#     now = datetime.now()
#     if "tuần sau" in query:
#         return (now + relativedelta(weeks=1)).strftime("%Y-%m-%d")
#     elif "tháng tới" in query:
#         return (now + relativedelta(months=1)).strftime("%Y-%m")
#     return None

# def extract_time(query):
#     # Các mẫu regex
#     month_pattern = r"tháng\s*(\d{1,2})"
#     day_month_pattern = r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})"
#     date_pattern = r"(\d{1,2})[/-](\d{1,2})"

#     # Trích xuất
#     month = re.search(month_pattern, query)
#     day_month = re.search(day_month_pattern, query)
#     date = re.search(date_pattern, query)

#     now = datetime.now()
#     current_year = now.year

#     if day_month:
#         day = int(day_month.group(1))
#         month_num = int(day_month.group(2))
#         return datetime(current_year, month_num, day).strftime("%Y-%m-%d")
#     elif month:
#         month_num = int(month.group(1))
#         return datetime(current_year, month_num, 1).strftime("%Y-%m")
#     elif date:
#         day = int(date.group(1))
#         month_num = int(date.group(2))
#         return datetime(current_year, month_num, day).strftime("%Y-%m-%d")
#     return None

# # Kết hợp hai hàm thành một hàm duy nhất
# def extract_all_times(query):
#     # Thử với extract_time trước
#     result = extract_time(query)
#     if result:
#         return result
#     # Nếu không, thử với parse_relative_time
#     result = parse_relative_time(query)
#     if result:
#         return result
#     return None

# # Test
# queries = [
#     "tôi muốn đặt tour vào tháng 9",
#     "tôi muốn đặt tour đi xuất phát từ ngày 3 tháng 5",
#     "tôi muốn đi vào 3/5",
#     "tôi muốn đặt tour tuần sau",
#     "tôi muốn đặt tour tháng tới"
# ]

# for q in queries:
#     print(f"Query: {q} -> Thời gian: {extract_all_times(q)}")