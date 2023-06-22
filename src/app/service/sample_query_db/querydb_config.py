
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext

'''
# Hàm này đang bị lỗi cần debug thêm

def get_password():
    # --- Get Item in Sharepoint list
    sp_client_id = '52fdd257-fd3e-4f56-86ca-3d5c7091c110'
    sp_client_secret = 'z+pRoP0UCRZBggLALjJ2Yw/GsBo8tuZF0A4qbKK3XaI='
    sp_client_credentials = ClientCredential(sp_client_id, sp_client_secret)
    sp_site_url = 'https://viendaukhivn.sharepoint.com/sites/VPIDataAnalytics30-RefreshPassword'
    ctx = ClientContext(sp_site_url).with_credentials(sp_client_credentials)
    target_list = ctx.web.lists.get_by_title("password_storage")
    target_item = target_list.get_item_by_id("1")

    # --- Get current_password
    ctx.load(target_item)
    ctx.execute_query()
    item_value = target_item.properties
    password = str(item_value["current_password"])
    return password
'''

# Sử dụng hàm sau để lấy password trực tiếp
def get_password():
    password = 'Muz42633'
    return password

username = 'api@oilgas.ai'
driver = '{ODBC Driver 18 for SQL Server}'

map_db = {
    "crudeOil": {
        "server": "xznozrobo3funm76yoyaoh75wm-lvvgvquleiuurnfvyvnetw7hoq.datamart.pbidedicated.windows.net",
        "database": "Oil price forecast"
    },
    "lpg": {
        "server": "xznozrobo3funm76yoyaoh75wm-fr3e3p3dk6eejffi7w4p27iybe.datamart.pbidedicated.windows.net",
        "database": "2023_LPG_Datamart_Hanhdh"
    },
    "hydrogen": {
        "server": "xznozrobo3funm76yoyaoh75wm-bskk54c73wdejgsv4xf2kugg5i.datamart.pbidedicated.windows.net",
        "database": "Global_Hydrogen_Data"
    },
    "crudeOilV2": {
        "server": "xznozrobo3funm76yoyaoh75wm-joiz6h43v2cuxennbhz3uklaa4.datamart.pbidedicated.windows.net",
        "database": "Crude Oil Price V2"
    },
    "gas": {
        "server": "xznozrobo3funm76yoyaoh75wm-hq556rblkuhevbjr3hywplbmcy.datamart.pbidedicated.windows.net",
        "database": "Domestic Gas Market_Production"
    },
    "shippingCost": {
        "server": "xznozrobo3funm76yoyaoh75wm-meed4hmxotsevljamvd2b3tp7i.datamart.pbidedicated.windows.net",
        "database": "LNG_Test"
    },
    "lantest": {
        "server": "xznozrobo3funm76yoyaoh75wm-qy262d2afrfuteltlticffpbb4.datamart.pbidedicated.windows.net",
        "database": "lantvh_epl"
    }
}

error_message = {
    "missing_product": "Missing product. Please check your product again",
    "product_not_found": "Product not found. Please check your product again",
    "table_not_found": "Table not found. Please check your table name again"
}


