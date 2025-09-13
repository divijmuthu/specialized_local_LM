"""
Actionable Recommendations Engine for Marketing, Pricing, and Supply Chain
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize, linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from config import Config

logger = logging.getLogger(__name__)

class MarketingRecommendationEngine:
    """Marketing recommendations based on customer segmentation and behavior analysis"""
    
    def __init__(self):
        self.customer_segments = None
        self.segment_characteristics = {}
        self.marketing_strategies = {}
    
    def analyze_customer_segments(self, customer_data: pd.DataFrame, segments: np.ndarray) -> Dict[str, Any]:
        """Analyze customer segments for marketing insights"""
        customer_data_with_segments = customer_data.copy()
        customer_data_with_segments['segment'] = segments
        
        segment_analysis = {}
        
        for segment in np.unique(segments):
            segment_data = customer_data_with_segments[customer_data_with_segments['segment'] == segment]
            
            analysis = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(customer_data) * 100,
                'avg_spend': segment_data['total_spend'].mean() if 'total_spend' in segment_data.columns else 0,
                'avg_frequency': segment_data['purchase_count'].mean() if 'purchase_count' in segment_data.columns else 0,
                'avg_recency': segment_data['days_since_last_purchase'].mean() if 'days_since_last_purchase' in segment_data.columns else 0,
                'top_products': segment_data['favorite_product'].value_counts().head(3).to_dict() if 'favorite_product' in segment_data.columns else {},
                'demographics': {
                    'avg_age': segment_data['age'].mean() if 'age' in segment_data.columns else 0,
                    'gender_distribution': segment_data['gender'].value_counts().to_dict() if 'gender' in segment_data.columns else {},
                    'location_distribution': segment_data['location'].value_counts().head(3).to_dict() if 'location' in segment_data.columns else {}
                }
            }
            
            segment_analysis[f'segment_{segment}'] = analysis
        
        self.segment_characteristics = segment_analysis
        return segment_analysis
    
    def generate_marketing_recommendations(self, customer_data: pd.DataFrame, segments: np.ndarray) -> Dict[str, List[str]]:
        """Generate marketing recommendations for each segment"""
        segment_analysis = self.analyze_customer_segments(customer_data, segments)
        recommendations = {}
        
        for segment_name, analysis in segment_analysis.items():
            segment_recommendations = []
            
            # High-value customer recommendations
            if analysis['avg_spend'] > customer_data['total_spend'].quantile(0.75):
                segment_recommendations.extend([
                    "Implement VIP customer program with exclusive benefits",
                    "Offer premium products and services",
                    "Provide personalized customer service",
                    "Create loyalty rewards program with higher tiers"
                ])
            
            # High-frequency customer recommendations
            if analysis['avg_frequency'] > customer_data['purchase_count'].quantile(0.75):
                segment_recommendations.extend([
                    "Implement subscription-based offerings",
                    "Create automated reorder reminders",
                    "Offer bulk purchase discounts",
                    "Develop habit-forming product bundles"
                ])
            
            # Low-frequency customer recommendations
            if analysis['avg_frequency'] < customer_data['purchase_count'].quantile(0.25):
                segment_recommendations.extend([
                    "Launch re-engagement email campaigns",
                    "Offer first-time buyer discounts",
                    "Implement referral programs",
                    "Create seasonal promotional campaigns"
                ])
            
            # High recency (recent customers) recommendations
            if analysis['avg_recency'] < customer_data['days_since_last_purchase'].quantile(0.25):
                segment_recommendations.extend([
                    "Send welcome series emails",
                    "Offer complementary product recommendations",
                    "Implement onboarding programs",
                    "Create early engagement surveys"
                ])
            
            # Low recency (dormant customers) recommendations
            if analysis['avg_recency'] > customer_data['days_since_last_purchase'].quantile(0.75):
                segment_recommendations.extend([
                    "Launch win-back campaigns with special offers",
                    "Implement customer feedback surveys",
                    "Offer reactivation discounts",
                    "Create personalized comeback incentives"
                ])
            
            # Age-based recommendations
            if analysis['demographics']['avg_age'] > 50:
                segment_recommendations.extend([
                    "Focus on traditional marketing channels",
                    "Emphasize product quality and reliability",
                    "Implement senior-friendly customer service",
                    "Create family-oriented product bundles"
                ])
            elif analysis['demographics']['avg_age'] < 30:
                segment_recommendations.extend([
                    "Leverage social media marketing",
                    "Implement influencer partnerships",
                    "Create mobile-first experiences",
                    "Offer trendy and innovative products"
                ])
            
            # Product-based recommendations
            if analysis['top_products']:
                top_product = max(analysis['top_products'], key=analysis['top_products'].get)
                segment_recommendations.extend([
                    f"Create cross-selling campaigns for {top_product}",
                    f"Develop {top_product} subscription services",
                    f"Launch {top_product} loyalty programs",
                    f"Offer {top_product} bundle deals"
                ])
            
            recommendations[segment_name] = list(set(segment_recommendations))
        
        self.marketing_strategies = recommendations
        return recommendations
    
    def calculate_marketing_roi(self, segment_recommendations: Dict[str, List[str]], 
                              customer_data: pd.DataFrame, segments: np.ndarray) -> Dict[str, float]:
        """Calculate potential ROI for marketing recommendations"""
        roi_estimates = {}
        
        for segment_name, recommendations in segment_recommendations.items():
            segment_num = int(segment_name.split('_')[1])
            segment_customers = customer_data[segments == segment_num]
            
            if len(segment_customers) == 0:
                continue
            
            # Estimate costs and benefits
            estimated_cost_per_customer = 10  # Base marketing cost
            estimated_conversion_rate = 0.15  # 15% conversion rate
            avg_customer_value = segment_customers['total_spend'].mean() if 'total_spend' in segment_customers.columns else 100
            
            total_customers = len(segment_customers)
            total_cost = total_customers * estimated_cost_per_customer
            expected_conversions = total_customers * estimated_conversion_rate
            expected_revenue = expected_conversions * avg_customer_value
            
            roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0
            roi_estimates[segment_name] = roi
        
        return roi_estimates

class PricingRecommendationEngine:
    """Pricing optimization recommendations"""
    
    def __init__(self):
        self.price_elasticity = {}
        self.optimal_prices = {}
        self.competitor_analysis = {}
    
    def calculate_price_elasticity(self, sales_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price elasticity for different products"""
        elasticity = {}
        
        for product in sales_data['product_id'].unique():
            product_data = sales_data[sales_data['product_id'] == product]
            
            if len(product_data) < 10:  # Need sufficient data
                continue
            
            # Calculate price elasticity using log-log regression
            prices = product_data['price'].values
            quantities = product_data['quantity'].values
            
            # Remove zeros and negative values
            mask = (prices > 0) & (quantities > 0)
            if np.sum(mask) < 5:
                continue
            
            prices = prices[mask]
            quantities = quantities[mask]
            
            # Log transformation
            log_prices = np.log(prices)
            log_quantities = np.log(quantities)
            
            # Calculate elasticity
            if len(log_prices) > 1:
                correlation = np.corrcoef(log_prices, log_quantities)[0, 1]
                price_std = np.std(log_prices)
                quantity_std = np.std(log_quantities)
                
                if price_std > 0:
                    elasticity[product] = correlation * (quantity_std / price_std)
        
        self.price_elasticity = elasticity
        return elasticity
    
    def optimize_pricing_strategy(self, products: List[str], 
                                current_prices: Dict[str, float],
                                costs: Dict[str, float],
                                demand_forecasts: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Optimize pricing strategy for maximum revenue"""
        pricing_recommendations = {}
        
        for product in products:
            if product not in current_prices or product not in costs:
                continue
            
            current_price = current_prices[product]
            cost = costs[product]
            elasticity = self.price_elasticity.get(product, -1.5)  # Default elasticity
            
            # Calculate optimal price using elasticity
            optimal_price = cost / (1 + 1/abs(elasticity)) if elasticity != 0 else current_price
            
            # Ensure price is within reasonable bounds
            min_price = cost * 1.1  # 10% markup minimum
            max_price = current_price * 2  # Maximum 2x current price
            
            optimal_price = max(min_price, min(optimal_price, max_price))
            
            # Calculate potential impact
            price_change = (optimal_price - current_price) / current_price
            
            # Estimate demand change based on elasticity
            demand_change = elasticity * price_change
            new_demand = demand_forecasts.get(product, 1000) * (1 + demand_change)
            
            # Calculate revenue impact
            current_revenue = current_price * demand_forecasts.get(product, 1000)
            new_revenue = optimal_price * new_demand
            
            pricing_recommendations[product] = {
                'current_price': current_price,
                'optimal_price': optimal_price,
                'price_change_percent': price_change * 100,
                'demand_change_percent': demand_change * 100,
                'revenue_impact': new_revenue - current_revenue,
                'revenue_impact_percent': (new_revenue - current_revenue) / current_revenue * 100,
                'elasticity': elasticity,
                'recommendation': self._get_pricing_recommendation(price_change, elasticity)
            }
        
        return pricing_recommendations
    
    def _get_pricing_recommendation(self, price_change: float, elasticity: float) -> str:
        """Get pricing recommendation based on analysis"""
        if abs(price_change) < 0.05:  # Less than 5% change
            return "Maintain current pricing strategy"
        elif price_change > 0.1:  # Significant price increase
            if abs(elasticity) < 1:  # Inelastic demand
                return "Implement price increase - demand is inelastic"
            else:
                return "Consider gradual price increase with demand monitoring"
        elif price_change < -0.1:  # Significant price decrease
            if abs(elasticity) > 1:  # Elastic demand
                return "Implement price decrease to increase volume"
            else:
                return "Consider promotional pricing instead of permanent decrease"
        else:
            return "Minor price adjustment recommended"
    
    def generate_dynamic_pricing_recommendations(self, 
                                               sales_data: pd.DataFrame,
                                               inventory_data: pd.DataFrame,
                                               competitor_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate dynamic pricing recommendations"""
        recommendations = {}
        
        for product in sales_data['product_id'].unique():
            product_recommendations = []
            product_sales = sales_data[sales_data['product_id'] == product]
            product_inventory = inventory_data[inventory_data['product_id'] == product] if not inventory_data.empty else pd.DataFrame()
            product_competitors = competitor_data[competitor_data['product_id'] == product] if not competitor_data.empty else pd.DataFrame()
            
            # Inventory-based recommendations
            if not product_inventory.empty:
                current_inventory = product_inventory['current_stock'].iloc[0] if 'current_stock' in product_inventory.columns else 0
                avg_daily_sales = product_sales['quantity'].mean() if not product_sales.empty else 0
                
                if current_inventory < avg_daily_sales * 7:  # Less than 1 week of inventory
                    product_recommendations.append("Increase price due to low inventory")
                elif current_inventory > avg_daily_sales * 30:  # More than 1 month of inventory
                    product_recommendations.append("Decrease price to clear excess inventory")
            
            # Sales trend recommendations
            if len(product_sales) > 7:
                recent_sales = product_sales.tail(7)['quantity'].mean()
                older_sales = product_sales.head(-7)['quantity'].mean() if len(product_sales) > 14 else recent_sales
                
                if recent_sales > older_sales * 1.2:  # 20% increase in sales
                    product_recommendations.append("Consider price increase due to increased demand")
                elif recent_sales < older_sales * 0.8:  # 20% decrease in sales
                    product_recommendations.append("Consider price decrease to stimulate demand")
            
            # Competitor-based recommendations
            if not product_competitors.empty:
                our_price = product_sales['price'].mean() if not product_sales.empty else 0
                competitor_avg_price = product_competitors['price'].mean()
                
                if our_price > competitor_avg_price * 1.1:  # 10% higher than competitors
                    product_recommendations.append("Consider price reduction to match competitors")
                elif our_price < competitor_avg_price * 0.9:  # 10% lower than competitors
                    product_recommendations.append("Consider price increase to improve margins")
            
            # Seasonal recommendations
            if 'date' in product_sales.columns:
                product_sales['month'] = pd.to_datetime(product_sales['date']).dt.month
                monthly_sales = product_sales.groupby('month')['quantity'].mean()
                
                if len(monthly_sales) > 1:
                    peak_month = monthly_sales.idxmax()
                    current_month = datetime.now().month
                    
                    if current_month == peak_month:
                        product_recommendations.append("Implement premium pricing during peak season")
                    elif abs(current_month - peak_month) <= 1:
                        product_recommendations.append("Prepare for seasonal price adjustments")
            
            recommendations[product] = product_recommendations
        
        return recommendations

class SupplyChainRecommendationEngine:
    """Supply chain optimization recommendations"""
    
    def __init__(self):
        self.supplier_performance = {}
        self.inventory_optimization = {}
        self.logistics_recommendations = {}
    
    def analyze_supplier_performance(self, supplier_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze supplier performance metrics"""
        supplier_analysis = {}
        
        for supplier in supplier_data['supplier_id'].unique():
            supplier_info = supplier_data[supplier_data['supplier_id'] == supplier]
            
            analysis = {
                'on_time_delivery_rate': (supplier_info['on_time'] == True).mean() if 'on_time' in supplier_info.columns else 0,
                'quality_score': supplier_info['quality_rating'].mean() if 'quality_rating' in supplier_info.columns else 0,
                'avg_cost': supplier_info['cost'].mean() if 'cost' in supplier_info.columns else 0,
                'avg_lead_time': supplier_info['lead_time'].mean() if 'lead_time' in supplier_info.columns else 0,
                'total_orders': len(supplier_info),
                'reliability_score': 0  # Will be calculated
            }
            
            # Calculate reliability score
            reliability_factors = [
                analysis['on_time_delivery_rate'],
                analysis['quality_score'] / 5 if analysis['quality_score'] > 0 else 0,  # Assuming 5-point scale
                1 - (analysis['avg_lead_time'] / 30) if analysis['avg_lead_time'] > 0 else 1  # Lead time factor
            ]
            analysis['reliability_score'] = np.mean(reliability_factors)
            
            supplier_analysis[supplier] = analysis
        
        self.supplier_performance = supplier_analysis
        return supplier_analysis
    
    def generate_supplier_recommendations(self, supplier_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate supplier management recommendations"""
        recommendations = {}
        
        for supplier, analysis in supplier_analysis.items():
            supplier_recommendations = []
            
            # On-time delivery recommendations
            if analysis['on_time_delivery_rate'] < 0.8:
                supplier_recommendations.extend([
                    "Implement stricter delivery penalties",
                    "Require delivery guarantees in contracts",
                    "Consider backup supplier options",
                    "Review supplier capacity and resources"
                ])
            elif analysis['on_time_delivery_rate'] > 0.95:
                supplier_recommendations.extend([
                    "Increase order volume with this supplier",
                    "Negotiate better pricing due to excellent performance",
                    "Consider exclusive partnership opportunities"
                ])
            
            # Quality recommendations
            if analysis['quality_score'] < 3.0:
                supplier_recommendations.extend([
                    "Implement quality improvement programs",
                    "Increase quality inspections",
                    "Consider supplier replacement",
                    "Require quality certifications"
                ])
            elif analysis['quality_score'] > 4.5:
                supplier_recommendations.extend([
                    "Reduce quality inspection frequency",
                    "Consider premium pricing for high quality",
                    "Use as preferred supplier for critical products"
                ])
            
            # Cost optimization recommendations
            if analysis['avg_cost'] > np.mean([a['avg_cost'] for a in supplier_analysis.values()]) * 1.2:
                supplier_recommendations.extend([
                    "Negotiate better pricing terms",
                    "Consider volume discounts",
                    "Evaluate alternative suppliers",
                    "Review cost structure and margins"
                ])
            
            # Lead time recommendations
            if analysis['avg_lead_time'] > 14:  # More than 2 weeks
                supplier_recommendations.extend([
                    "Negotiate shorter lead times",
                    "Implement just-in-time inventory",
                    "Consider local supplier alternatives",
                    "Optimize order frequency and quantities"
                ])
            
            # Overall reliability recommendations
            if analysis['reliability_score'] < 0.6:
                supplier_recommendations.extend([
                    "Develop supplier improvement plan",
                    "Consider supplier replacement",
                    "Implement risk mitigation strategies",
                    "Increase monitoring and reporting"
                ])
            elif analysis['reliability_score'] > 0.9:
                supplier_recommendations.extend([
                    "Consider strategic partnership",
                    "Increase business volume",
                    "Use as primary supplier for critical items"
                ])
            
            recommendations[supplier] = list(set(supplier_recommendations))
        
        return recommendations
    
    def optimize_inventory_levels(self, demand_data: pd.DataFrame, 
                                inventory_data: pd.DataFrame,
                                lead_time_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Optimize inventory levels using demand forecasting and lead times"""
        inventory_recommendations = {}
        
        for product in demand_data['product_id'].unique():
            product_demand = demand_data[demand_data['product_id'] == product]
            product_inventory = inventory_data[inventory_data['product_id'] == product] if not inventory_data.empty else pd.DataFrame()
            product_lead_time = lead_time_data[lead_time_data['product_id'] == product] if not lead_time_data.empty else pd.DataFrame()
            
            # Calculate demand statistics
            avg_daily_demand = product_demand['demand'].mean() if not product_demand.empty else 0
            demand_std = product_demand['demand'].std() if not product_demand.empty else 0
            avg_lead_time = product_lead_time['lead_time'].mean() if not product_lead_time.empty else 7
            
            # Calculate optimal inventory levels
            safety_stock = 1.96 * demand_std * np.sqrt(avg_lead_time)  # 95% service level
            reorder_point = avg_daily_demand * avg_lead_time + safety_stock
            optimal_order_quantity = np.sqrt(2 * avg_daily_demand * 365 * 50 / 10)  # EOQ formula (simplified)
            
            current_inventory = product_inventory['current_stock'].iloc[0] if not product_inventory.empty and 'current_stock' in product_inventory.columns else 0
            
            inventory_recommendations[product] = {
                'current_inventory': current_inventory,
                'optimal_inventory': reorder_point,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'optimal_order_quantity': optimal_order_quantity,
                'avg_daily_demand': avg_daily_demand,
                'avg_lead_time': avg_lead_time,
                'inventory_status': self._get_inventory_status(current_inventory, reorder_point),
                'recommendations': self._get_inventory_recommendations(current_inventory, reorder_point, optimal_order_quantity)
            }
        
        return inventory_recommendations
    
    def _get_inventory_status(self, current_inventory: float, reorder_point: float) -> str:
        """Get inventory status"""
        if current_inventory <= reorder_point * 0.5:
            return "Critical - Immediate reorder needed"
        elif current_inventory <= reorder_point:
            return "Low - Reorder recommended"
        elif current_inventory <= reorder_point * 1.5:
            return "Adequate - Monitor closely"
        else:
            return "High - Consider reducing orders"
    
    def _get_inventory_recommendations(self, current_inventory: float, reorder_point: float, optimal_order_quantity: float) -> List[str]:
        """Get inventory management recommendations"""
        recommendations = []
        
        if current_inventory <= reorder_point * 0.5:
            recommendations.extend([
                "Place urgent reorder immediately",
                "Consider expedited shipping",
                "Review demand forecasting accuracy",
                "Implement safety stock increase"
            ])
        elif current_inventory <= reorder_point:
            recommendations.extend([
                "Place standard reorder",
                "Review lead time optimization",
                "Consider supplier performance"
            ])
        elif current_inventory > reorder_point * 2:
            recommendations.extend([
                "Reduce future order quantities",
                "Consider promotional activities",
                "Review demand forecasting",
                "Optimize inventory turnover"
            ])
        
        return recommendations
    
    def optimize_logistics(self, shipping_data: pd.DataFrame, 
                         warehouse_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Optimize logistics and distribution"""
        logistics_recommendations = {}
        
        # Shipping optimization
        if not shipping_data.empty:
            avg_shipping_cost = shipping_data['shipping_cost'].mean()
            avg_delivery_time = shipping_data['delivery_time'].mean()
            
            logistics_recommendations['shipping'] = []
            
            if avg_shipping_cost > shipping_data['shipping_cost'].quantile(0.75):
                logistics_recommendations['shipping'].extend([
                    "Negotiate better shipping rates",
                    "Consider bulk shipping discounts",
                    "Evaluate alternative carriers",
                    "Implement shipping cost optimization"
                ])
            
            if avg_delivery_time > shipping_data['delivery_time'].quantile(0.75):
                logistics_recommendations['shipping'].extend([
                    "Optimize delivery routes",
                    "Consider regional distribution centers",
                    "Implement faster shipping options",
                    "Review carrier performance"
                ])
        
        # Warehouse optimization
        if not warehouse_data.empty:
            warehouse_utilization = warehouse_data['utilization_rate'].mean() if 'utilization_rate' in warehouse_data.columns else 0
            
            logistics_recommendations['warehouse'] = []
            
            if warehouse_utilization > 0.9:
                logistics_recommendations['warehouse'].extend([
                    "Consider warehouse expansion",
                    "Optimize storage layout",
                    "Implement automated systems",
                    "Review inventory placement"
                ])
            elif warehouse_utilization < 0.6:
                logistics_recommendations['warehouse'].extend([
                    "Consolidate warehouse operations",
                    "Sublet unused space",
                    "Optimize warehouse layout",
                    "Review space utilization"
                ])
        
        return logistics_recommendations

class RecommendationEngine:
    """Main recommendation engine that combines all recommendation types"""
    
    def __init__(self):
        self.marketing_engine = MarketingRecommendationEngine()
        self.pricing_engine = PricingRecommendationEngine()
        self.supply_chain_engine = SupplyChainRecommendationEngine()
    
    def generate_comprehensive_recommendations(self, 
                                             customer_data: pd.DataFrame,
                                             sales_data: pd.DataFrame,
                                             inventory_data: pd.DataFrame,
                                             supplier_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive business recommendations"""
        
        recommendations = {
            'marketing': {},
            'pricing': {},
            'supply_chain': {},
            'summary': {}
        }
        
        # Marketing recommendations
        if not customer_data.empty:
            # Perform customer segmentation
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            numeric_features = customer_data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_features:
                X = customer_data[numeric_features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=5, random_state=42)
                segments = kmeans.fit_predict(X_scaled)
                
                # Generate marketing recommendations
                marketing_recs = self.marketing_engine.generate_marketing_recommendations(customer_data, segments)
                roi_estimates = self.marketing_engine.calculate_marketing_roi(marketing_recs, customer_data, segments)
                
                recommendations['marketing'] = {
                    'segment_recommendations': marketing_recs,
                    'roi_estimates': roi_estimates,
                    'segment_analysis': self.marketing_engine.segment_characteristics
                }
        
        # Pricing recommendations
        if not sales_data.empty:
            # Calculate price elasticity
            elasticity = self.pricing_engine.calculate_price_elasticity(sales_data)
            
            # Generate pricing recommendations
            products = sales_data['product_id'].unique()
            current_prices = sales_data.groupby('product_id')['price'].mean().to_dict()
            costs = {p: current_prices[p] * 0.6 for p in products}  # Assume 40% margin
            demand_forecasts = sales_data.groupby('product_id')['quantity'].mean().to_dict()
            
            pricing_recs = self.pricing_engine.optimize_pricing_strategy(products, current_prices, costs, demand_forecasts)
            dynamic_pricing_recs = self.pricing_engine.generate_dynamic_pricing_recommendations(sales_data, inventory_data, pd.DataFrame())
            
            recommendations['pricing'] = {
                'price_optimization': pricing_recs,
                'dynamic_pricing': dynamic_pricing_recs,
                'elasticity_analysis': elasticity
            }
        
        # Supply chain recommendations
        if supplier_data is not None and not supplier_data.empty:
            supplier_analysis = self.supply_chain_engine.analyze_supplier_performance(supplier_data)
            supplier_recs = self.supply_chain_engine.generate_supplier_recommendations(supplier_analysis)
            
            recommendations['supply_chain']['supplier_recommendations'] = supplier_recs
            recommendations['supply_chain']['supplier_analysis'] = supplier_analysis
        
        # Inventory optimization
        if not inventory_data.empty and not sales_data.empty:
            # Create demand data from sales
            demand_data = sales_data.groupby(['product_id', 'date'])['quantity'].sum().reset_index()
            demand_data.rename(columns={'quantity': 'demand'}, inplace=True)
            
            inventory_recs = self.supply_chain_engine.optimize_inventory_levels(demand_data, inventory_data, pd.DataFrame())
            logistics_recs = self.supply_chain_engine.optimize_logistics(pd.DataFrame(), pd.DataFrame())
            
            recommendations['supply_chain']['inventory_optimization'] = inventory_recs
            recommendations['supply_chain']['logistics_optimization'] = logistics_recs
        
        # Generate summary recommendations
        recommendations['summary'] = self._generate_summary_recommendations(recommendations)
        
        return recommendations
    
    def _generate_summary_recommendations(self, recommendations: Dict[str, Any]) -> List[str]:
        """Generate high-level summary recommendations"""
        summary = []
        
        # Marketing summary
        if 'marketing' in recommendations and recommendations['marketing']:
            if 'roi_estimates' in recommendations['marketing']:
                best_roi_segment = max(recommendations['marketing']['roi_estimates'], 
                                     key=recommendations['marketing']['roi_estimates'].get)
                summary.append(f"Focus marketing efforts on {best_roi_segment} for highest ROI")
        
        # Pricing summary
        if 'pricing' in recommendations and recommendations['pricing']:
            if 'price_optimization' in recommendations['pricing']:
                total_revenue_impact = sum(rec.get('revenue_impact', 0) for rec in recommendations['pricing']['price_optimization'].values())
                if total_revenue_impact > 0:
                    summary.append(f"Implement pricing optimization for potential revenue increase of ${total_revenue_impact:,.2f}")
        
        # Supply chain summary
        if 'supply_chain' in recommendations and recommendations['supply_chain']:
            if 'inventory_optimization' in recommendations['supply_chain']:
                critical_items = [item for item, data in recommendations['supply_chain']['inventory_optimization'].items() 
                                if data['inventory_status'].startswith('Critical')]
                if critical_items:
                    summary.append(f"Urgent inventory reorder needed for {len(critical_items)} products")
        
        return summary
    
    def save_recommendations(self, recommendations: Dict[str, Any], output_path: str):
        """Save recommendations to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"Recommendations saved to {output_path}")
    
    def load_recommendations(self, input_path: str) -> Dict[str, Any]:
        """Load recommendations from file"""
        with open(input_path, 'r') as f:
            recommendations = json.load(f)
        
        logger.info(f"Recommendations loaded from {input_path}")
        return recommendations

if __name__ == "__main__":
    # Test the recommendation engine
    from pipeline.data_collectors import SampleDataGenerator
    
    # Generate sample data
    generator = SampleDataGenerator()
    sample_data = {
        'customers': generator.generate_customer_data(1000),
        'sales': generator.generate_sales_data(2000),
        'feedback': generator.generate_feedback_data(1000)
    }
    
    # Create sample inventory data
    inventory_data = pd.DataFrame({
        'product_id': [f'PROD_{i:03d}' for i in range(1, 101)],
        'current_stock': np.random.randint(10, 1000, 100),
        'reorder_point': np.random.randint(50, 500, 100)
    })
    
    # Create sample supplier data
    supplier_data = pd.DataFrame({
        'supplier_id': [f'SUP_{i:03d}' for i in range(1, 21)],
        'on_time': np.random.choice([True, False], 20, p=[0.8, 0.2]),
        'quality_rating': np.random.uniform(2, 5, 20),
        'cost': np.random.uniform(10, 100, 20),
        'lead_time': np.random.randint(3, 21, 20)
    })
    
    # Generate recommendations
    engine = RecommendationEngine()
    recommendations = engine.generate_comprehensive_recommendations(
        sample_data['customers'],
        sample_data['sales'],
        inventory_data,
        supplier_data
    )
    
    print("Comprehensive Recommendations Generated:")
    print(f"Marketing segments: {len(recommendations['marketing'].get('segment_recommendations', {}))}")
    print(f"Pricing optimizations: {len(recommendations['pricing'].get('price_optimization', {}))}")
    print(f"Supply chain recommendations: {len(recommendations['supply_chain'])}")
    print(f"Summary recommendations: {len(recommendations['summary'])}")
    
    # Save recommendations
    engine.save_recommendations(recommendations, './data/recommendations.json')
    
    print("Recommendation engine test completed successfully!")
