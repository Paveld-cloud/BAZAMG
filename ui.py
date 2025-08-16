from telegram import InlineKeyboardMarkup, InlineKeyboardButton

def cancel_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]])

def confirm_markup():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Да, списать", callback_data="confirm_yes"),
         InlineKeyboardButton("❌ Нет", callback_data="confirm_no")],
        [InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]
    ])

def more_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("⏭ Ещё", callback_data="more")]])

def main_menu_markup():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Поиск", callback_data="menu_search")],
        [InlineKeyboardButton("📦 Как списать деталь", callback_data="menu_issue_help")],
        [InlineKeyboardButton("📞 Поддержка", callback_data="menu_contact")],
    ])