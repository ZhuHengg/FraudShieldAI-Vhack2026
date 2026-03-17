export const formatCurrency = (amount, currency = 'MYR') => {
  return new Intl.NumberFormat('en-MY', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2
  }).format(amount).replace(currency, currency + ' ')
}

export const formatTime = (dateInput) => {
  const date = typeof dateInput === 'string' ? new Date(dateInput) : dateInput
  return new Intl.DateTimeFormat('en-MY', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  }).format(date)
}

export const formatScore = (score) => {
  return (score * 100).toFixed(0) + '%'
}
