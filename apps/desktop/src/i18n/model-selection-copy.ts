/** Recovery copy for an eligibility set that changed before Apply. */
export const modelUnavailableCopy: Record<string, string> = {
  en: 'This model is no longer available. Choose another model or refresh the list.',
  zh: '此模型已不可用。请选择其他模型或刷新列表。',
  'zh-hant': '此模型已不可用。請選擇其他模型或重新整理清單。',
  ja: 'このモデルは利用できなくなりました。別のモデルを選ぶか、一覧を更新してください。',
  ar: 'لم يعد هذا النموذج متاحًا. اختر نموذجًا آخر أو حدّث القائمة.',
  ru: 'Эта модель больше недоступна. Выберите другую модель или обновите список.',
  fr: 'Ce modèle n’est plus disponible. Choisissez un autre modèle ou actualisez la liste.',
  de: 'Dieses Modell ist nicht mehr verfügbar. Wählen Sie ein anderes Modell oder aktualisieren Sie die Liste.',
  es: 'Este modelo ya no está disponible. Elige otro modelo o actualiza la lista.'
}

/** Guidance for a saved model outside the current eligible set. */
export const savedModelUnavailableCopy: Record<string, string> = {
  en: 'This saved model is unavailable. Choose an available model.',
  zh: '保存的模型不可用。请选择可用的模型。',
  'zh-hant': '儲存的模型無法使用。請選擇可用的模型。',
  ja: '保存されたモデルは利用できません。利用可能なモデルを選択してください。',
  ar: 'النموذج المحفوظ غير متاح. اختر نموذجًا متاحًا.',
  ru: 'Сохранённая модель недоступна. Выберите доступную модель.',
  fr: 'Le modèle enregistré est indisponible. Choisissez un modèle disponible.',
  de: 'Das gespeicherte Modell ist nicht verfügbar. Wählen Sie ein verfügbares Modell.',
  es: 'El modelo guardado no está disponible. Elige un modelo disponible.'
}

/** Unknown is not an enabled or disabled model list. */
export const modelEligibilityUnknownCopy: Record<string, string> = {
  en: 'Model availability could not be verified. Refresh the list or check the provider configuration.',
  zh: '无法确认模型是否可用。请刷新列表或检查服务商配置。',
  'zh-hant': '無法確認模型是否可用。請重新整理清單或檢查供應商設定。',
  ja: 'モデルの利用可否を確認できませんでした。一覧を更新するか、プロバイダーの設定を確認してください。',
  ar: 'تعذر التحقق من توفر النماذج. حدّث القائمة أو تحقق من إعدادات المزود.',
  ru: 'Не удалось проверить доступность моделей. Обновите список или проверьте настройки провайдера.',
  fr: 'Impossible de vérifier la disponibilité des modèles. Actualisez la liste ou vérifiez la configuration du fournisseur.',
  de: 'Die Verfügbarkeit der Modelle konnte nicht geprüft werden. Aktualisieren Sie die Liste oder prüfen Sie die Anbieter-Konfiguration.',
  es: 'No se pudo verificar la disponibilidad de los modelos. Actualiza la lista o revisa la configuración del proveedor.'
}
