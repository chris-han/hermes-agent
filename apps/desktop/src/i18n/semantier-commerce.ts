/** Native commerce labels use the application's current locale. */
interface CommerceCopy {
  merchant: string; currency: string; term: string; annual: string; monthly: string
  package: string; noPackage: string; active: string; expired: string; expires: string
  renewal: string; manual: string; allowance: string; purchased: string; environment: string
  manage: string; plans: string; topUp: string; orders: string; browserAuth: string
}

const copy = (values: string[]): CommerceCopy => Object.fromEntries(
  ['package', 'noPackage', 'active', 'expired', 'expires', 'renewal', 'manual', 'allowance', 'purchased', 'environment', 'manage', 'plans', 'topUp', 'orders', 'browserAuth', 'merchant', 'currency', 'term', 'annual', 'monthly']
    .map((key, index) => [key, values[index]])
) as unknown as CommerceCopy

export const semantierCommerceCopy: Record<string, CommerceCopy> = {
  en: copy(['Package', 'No package', 'Active', 'Expired', 'Expires', 'Renewal', 'Manual renewal', 'Included allowance', 'Purchased credit', 'Environment', 'Manage billing', 'Choose / renew package', 'Top up', 'Orders', 'Continue in your browser. Sign in separately to manage billing; return here to refresh.', "Merchant", "Currency", "Term", "Annual", "Monthly"]),
  zh: copy(['套餐', '暂无套餐', '生效中', '已过期', '有效期至', '续费方式', '手动续费', '套餐内额度', '已购额度', '环境', '管理账单', '选择 / 续费套餐', '充值', '订单', '在浏览器中继续，管理账单需单独登录。返回此处后自动刷新。', "收款主体", "币种", "期限", "年度", "月度"]),
  'zh-hant': copy(['方案', '尚無方案', '使用中', '已到期', '有效期限', '續費方式', '手動續費', '方案額度', '已購額度', '環境', '管理帳單', '選擇 / 續費方案', '儲值', '訂單', '在瀏覽器中繼續，管理帳單需另外登入。返回後自動更新。', "收款主體", "幣別", "期限", "年度", "月度"]),
  ja: copy(['プラン', 'プランなし', '有効', '期限切れ', '有効期限', '更新方法', '手動更新', 'プラン内クレジット', '購入クレジット', '環境', '請求を管理', 'プラン選択 / 更新', 'チャージ', '注文', 'ブラウザーで続行します。請求管理には別途ログインが必要です。戻ると更新されます。', "販売者", "通貨", "期間", "年額", "月額"]),
  fr: copy(['Forfait', 'Aucun forfait', 'Actif', 'Expiré', 'Expiration', 'Renouvellement', 'Manuel', 'Crédit inclus', 'Crédit acheté', 'Environnement', 'Gérer la facturation', 'Choisir / renouveler', 'Recharger', 'Commandes', 'Continuez dans le navigateur. Connectez-vous séparément pour gérer la facturation ; le retour actualise les données.', "Vendeur", "Devise", "Durée", "Annuel", "Mensuel"]),
  de: copy(['Tarif', 'Kein Tarif', 'Aktiv', 'Abgelaufen', 'Gültig bis', 'Verlängerung', 'Manuell', 'Enthaltenes Guthaben', 'Gekauftes Guthaben', 'Umgebung', 'Abrechnung verwalten', 'Tarif wählen / verlängern', 'Aufladen', 'Bestellungen', 'Im Browser fortfahren. Zur Abrechnung separat anmelden; bei Rückkehr werden die Daten aktualisiert.', "Verkäufer", "Währung", "Laufzeit", "Jährlich", "Monatlich"]),
  es: copy(['Plan', 'Sin plan', 'Activo', 'Caducado', 'Vence', 'Renovación', 'Manual', 'Crédito incluido', 'Crédito comprado', 'Entorno', 'Gestionar facturación', 'Elegir / renovar plan', 'Recargar', 'Pedidos', 'Continúa en el navegador. Inicia sesión por separado para gestionar la facturación; se actualiza al volver.', "Vendedor", "Moneda", "Periodo", "Anual", "Mensual"]),
  ru: copy(['Тариф', 'Без тарифа', 'Активен', 'Истёк', 'Действует до', 'Продление', 'Вручную', 'Включённый кредит', 'Купленный кредит', 'Среда', 'Управление оплатой', 'Выбрать / продлить тариф', 'Пополнить', 'Заказы', 'Продолжите в браузере. Для оплаты войдите отдельно; при возврате данные обновятся.', "Продавец", "Валюта", "Срок", "Годовой", "Месячный"]),
  ar: copy(['الخطة', 'لا توجد خطة', 'نشطة', 'منتهية', 'تنتهي في', 'التجديد', 'يدوي', 'الرصيد المشمول', 'الرصيد المشترى', 'البيئة', 'إدارة الفوترة', 'اختيار / تجديد الخطة', 'إضافة رصيد', 'الطلبات', 'تابع في المتصفح وسجّل الدخول بشكل مستقل لإدارة الفوترة. تُحدّث البيانات عند العودة.', "التاجر", "العملة", "المدة", "سنوي", "شهري"])
}
