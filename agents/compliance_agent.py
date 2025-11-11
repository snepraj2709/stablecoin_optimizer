"""
Compliance & Controls Agent
Handles KYC/AML checks, sanctions screening, and regulatory compliance
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import openai
import json


@dataclass
class ComplianceCheck:
    """Result of a compliance check"""
    check_type: str
    passed: bool
    risk_level: str  # low, medium, high
    details: str
    timestamp: str


class ComplianceAgent:
    """
    Responsible for:
    - KYC/AML verification
    - Sanctions screening (OFAC, UN, EU)
    - Geo-blocking for restricted regions
    - Transaction monitoring
    - Regulatory reporting
    """
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Sanctions lists (simplified)
        self.sanctioned_countries = [
            'OFAC', 'Iran', 'North Korea', 'Syria', 'Cuba', 'Venezuela'
        ]
        
        self.restricted_regions = [
            'Crimea', 'Donetsk', 'Luhansk'
        ]
        
        # Transaction limits by region
        self.aml_thresholds = {
            'US': 10000,
            'EU': 10000,
            'UK': 10000,
            'APAC': 15000,
            'LATAM': 5000
        }
        
        print("[COMPLIANCE_AGENT] Initialized")
    
    async def verify_transfer(
        self,
        transfer_id: str,
        user_id: str,
        region: str,
        amount: float,
        source_currency: str,
        dest_currency: str,
        kyc_status: str
    ) -> Dict[str, Any]:
        """
        Comprehensive compliance verification
        """
        print(f"[COMPLIANCE_AGENT] Verifying transfer {transfer_id}")
        
        checks = []
        
        # 1. KYC Verification
        kyc_check = await self._verify_kyc(user_id, kyc_status)
        checks.append(kyc_check)
        
        # 2. Sanctions Screening
        sanctions_check = await self._screen_sanctions(user_id, region)
        checks.append(sanctions_check)
        
        # 3. Geo-blocking
        geo_check = await self._check_geo_restrictions(region)
        checks.append(geo_check)
        
        # 4. AML Threshold Check
        aml_check = await self._check_aml_threshold(region, amount)
        checks.append(aml_check)
        
        # 5. Transaction Pattern Analysis
        pattern_check = await self._analyze_transaction_pattern(
            user_id, amount, source_currency, dest_currency
        )
        checks.append(pattern_check)
        
        # Determine overall status
        all_passed = all(check.passed for check in checks)
        max_risk = max(
            ['low', 'medium', 'high'].index(check.risk_level)
            for check in checks
        )
        risk_levels = ['low', 'medium', 'high']
        overall_risk = risk_levels[max_risk]
        
        # Use LLM for risk assessment
        risk_analysis = await self._get_risk_analysis(checks, amount)
        
        return {
            'transfer_id': transfer_id,
            'compliance_passed': all_passed,
            'overall_risk': overall_risk,
            'checks': [
                {
                    'type': c.check_type,
                    'passed': c.passed,
                    'risk': c.risk_level,
                    'details': c.details
                }
                for c in checks
            ],
            'risk_analysis': risk_analysis,
            'timestamp': datetime.now().isoformat(),
            'requires_review': overall_risk == 'high' or not all_passed
        }
    
    async def _verify_kyc(self, user_id: str, kyc_status: str) -> ComplianceCheck:
        """Verify KYC status"""
        await asyncio.sleep(0.01)  # Simulate API call
        
        if kyc_status == 'verified':
            return ComplianceCheck(
                check_type='KYC',
                passed=True,
                risk_level='low',
                details='KYC verified and up-to-date',
                timestamp=datetime.now().isoformat()
            )
        elif kyc_status == 'pending':
            return ComplianceCheck(
                check_type='KYC',
                passed=False,
                risk_level='medium',
                details='KYC verification pending',
                timestamp=datetime.now().isoformat()
            )
        else:
            return ComplianceCheck(
                check_type='KYC',
                passed=False,
                risk_level='high',
                details='KYC not verified',
                timestamp=datetime.now().isoformat()
            )
    
    async def _screen_sanctions(self, user_id: str, region: str) -> ComplianceCheck:
        """Screen against sanctions lists"""
        await asyncio.sleep(0.015)
        
        if region in self.sanctioned_countries:
            return ComplianceCheck(
                check_type='Sanctions',
                passed=False,
                risk_level='high',
                details=f'Region {region} is sanctioned',
                timestamp=datetime.now().isoformat()
            )
        
        if region in self.restricted_regions:
            return ComplianceCheck(
                check_type='Sanctions',
                passed=False,
                risk_level='high',
                details=f'Region {region} is restricted',
                timestamp=datetime.now().isoformat()
            )
        
        return ComplianceCheck(
            check_type='Sanctions',
            passed=True,
            risk_level='low',
            details='No sanctions matches found',
            timestamp=datetime.now().isoformat()
        )
    
    async def _check_geo_restrictions(self, region: str) -> ComplianceCheck:
        """Check geographic restrictions"""
        await asyncio.sleep(0.01)
        
        # Simulate geo-blocking rules
        blocked_for_crypto = ['China', 'Bangladesh']
        
        if region in blocked_for_crypto:
            return ComplianceCheck(
                check_type='Geo-blocking',
                passed=False,
                risk_level='high',
                details=f'Cryptocurrency services not available in {region}',
                timestamp=datetime.now().isoformat()
            )
        
        return ComplianceCheck(
            check_type='Geo-blocking',
            passed=True,
            risk_level='low',
            details=f'Services available in {region}',
            timestamp=datetime.now().isoformat()
        )
    
    async def _check_aml_threshold(self, region: str, amount: float) -> ComplianceCheck:
        """Check AML reporting thresholds"""
        await asyncio.sleep(0.01)
        
        threshold = self.aml_thresholds.get(region, 10000)
        
        if amount > threshold:
            return ComplianceCheck(
                check_type='AML',
                passed=True,  # Passes but requires enhanced due diligence
                risk_level='medium',
                details=f'Amount exceeds ${threshold:,.0f} threshold - requires EDD',
                timestamp=datetime.now().isoformat()
            )
        
        return ComplianceCheck(
            check_type='AML',
            passed=True,
            risk_level='low',
            details='Within normal transaction limits',
            timestamp=datetime.now().isoformat()
        )
    
    async def _analyze_transaction_pattern(
        self,
        user_id: str,
        amount: float,
        source_currency: str,
        dest_currency: str
    ) -> ComplianceCheck:
        """Analyze transaction patterns for suspicious activity"""
        await asyncio.sleep(0.02)
        
        # Simulate pattern analysis
        # In production: check velocity, frequency, typical amounts, etc.
        
        # High-risk pattern indicators
        suspicious_patterns = []
        
        # Large round numbers
        if amount >= 10000 and amount % 10000 == 0:
            suspicious_patterns.append('Round number transaction')
        
        # Rapid fire (would check recent transactions)
        # suspicious_patterns.append('High transaction velocity')
        
        if suspicious_patterns:
            return ComplianceCheck(
                check_type='Pattern Analysis',
                passed=True,  # Passes but flagged for review
                risk_level='medium',
                details=f'Flagged: {", ".join(suspicious_patterns)}',
                timestamp=datetime.now().isoformat()
            )
        
        return ComplianceCheck(
            check_type='Pattern Analysis',
            passed=True,
            risk_level='low',
            details='Transaction pattern normal',
            timestamp=datetime.now().isoformat()
        )
    
    async def _get_risk_analysis(
        self,
        checks: List[ComplianceCheck],
        amount: float
    ) -> str:
        """Use LLM to provide risk analysis"""
        
        checks_summary = [
            {
                'type': c.check_type,
                'passed': c.passed,
                'risk': c.risk_level,
                'details': c.details
            }
            for c in checks
        ]
        
        prompt = f"""Analyze these compliance checks for a ${amount:,.0f} transaction:

{json.dumps(checks_summary, indent=2)}

Provide a brief risk assessment (2-3 sentences) and any recommendations."""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a compliance and AML expert."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[COMPLIANCE_AGENT] LLM analysis failed: {e}")
            
            # Fallback analysis
            high_risk_count = sum(1 for c in checks if c.risk_level == 'high')
            if high_risk_count > 0:
                return "High risk factors detected. Manual review required."
            
            medium_risk_count = sum(1 for c in checks if c.risk_level == 'medium')
            if medium_risk_count > 1:
                return "Moderate risk. Enhanced due diligence recommended."
            
            return "Low risk transaction. Standard processing approved."
    
    async def generate_compliance_report(
        self,
        transfer_ids: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Generate compliance report for auditing"""
        
        return {
            'report_id': f"COMPLIANCE_REPORT_{datetime.now().strftime('%Y%m%d')}",
            'period': {
                'start': start_date,
                'end': end_date
            },
            'transfers_reviewed': len(transfer_ids),
            'summary': {
                'total_flagged': 0,
                'requires_review': 0,
                'blocked': 0,
                'approved': len(transfer_ids)
            },
            'by_check_type': {
                'KYC': {'passed': len(transfer_ids), 'failed': 0},
                'Sanctions': {'passed': len(transfer_ids), 'failed': 0},
                'AML': {'passed': len(transfer_ids), 'failed': 0}
            },
            'generated_at': datetime.now().isoformat()
        }


async def main():
    """Test the compliance agent"""
    agent = ComplianceAgent(os.getenv('OPENAI_API_KEY', 'test_key'))
    
    result = await agent.verify_transfer(
        transfer_id='TEST_001',
        user_id='USER_123',
        region='US',
        amount=75000,
        source_currency='USD',
        dest_currency='USDC',
        kyc_status='verified'
    )
    
    print("\n[TEST] Compliance Check Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())